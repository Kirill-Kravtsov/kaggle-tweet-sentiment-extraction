import re
from typing import Dict, Tuple, Union
from pathlib import Path
from catalyst.core import Callback, MetricCallback, CallbackOrder, utils
from catalyst.core.callbacks import CheckpointCallback
from utils import jaccard_func, jaccard_func_dense, CustomSWA


class JaccardCallback(MetricCallback):

    def __init__(
        self,
        input_key=['start_positions', 'end_positions', 'orig_tweet', 'orig_selected', 'offsets'],
        output_key=['start_logits', 'end_logits'],
        prefix="jaccard",
        activation=None,
        dense=False
    ):
        if dense:
            super().__init__(
                prefix=prefix,
                metric_fn=jaccard_func_dense,
                input_key=input_key,
                output_key=output_key
                #activation=activation
            )
        else:
            super().__init__(
                prefix=prefix,
                metric_fn=jaccard_func,
                input_key=input_key,
                output_key=output_key
                #activation=activation
            )


class SWACallback(Callback):

    def __init__(self, swa_start=10, swa_freq=1, alpha=0.05, num_swap_epochs=1, swap_best=False):
        super().__init__(order=CallbackOrder.Internal)
        self.swa_start = swa_start
        self.swa_freq = swa_freq
        self.alpha = alpha
        self.swapped = False
        self.num_swap_epochs = num_swap_epochs
        self.swap_best = swap_best
        self.best = None

    def on_stage_start(self, state):
        state.optimizer = CustomSWA(
            optimizer=state.optimizer,
            swa_start=self.swa_start,
            swa_freq=self.swa_freq,
            alpha=self.alpha
        )

    def on_loader_start(self, state):
        """
        if (state.loader_name == "train") and state.epoch > 1:
            if self.swap_best:
                if self.best == "train"
            and state.epoch > (1 + self.num_swap_epochs):
            state.optimizer.swap_swa_sgd()
        """
        if state.loader_name == "valid_swa":
            state.optimizer.swap_swa_sgd()

    def on_epoch_end(self, state):
        if self.swap_best:
            train_jac = state.epoch_metrics['valid_jaccard']
            swa_jac = state.epoch_metrics['valid_swa_jaccard']
            # at this point swa weight are in the model
            if train_jac > swa_jac:
                # rewrite swa weights/loss/jaccard
                print("Rewriting swa")
                state.optimizer.update_model_weights()
                state.epoch_metrics['valid_swa_loss'] = state.epoch_metrics['valid_loss'] 
                state.epoch_metrics['valid_swa_jaccard'] = state.epoch_metrics['valid_jaccard'] 
            elif state.epoch < state.num_epochs:
                state.optimizer.swap_swa_sgd()

        elif state.epoch > self.num_swap_epochs:
            state.optimizer.swap_swa_sgd()



class FreezeControlCallback(Callback):

    def __init__(self, schedule, always_frozen=()):
        """
        Layers not found in schedule dict and always_frozen will be trainable
        """
        super().__init__(order=CallbackOrder.Internal)
        self.schedule = schedule 

        for k in self.schedule:
            if isinstance(self.schedule[k], str):
                self.schedule[k] = [v]
            self.schedule[k] = [re.compile(pattern) for pattern in self.schedule[k]]

        self.always_frozen = [re.compile(pattern) for pattern in always_frozen]
        

    def on_stage_start(self, state):
        for param, data in state.model.named_parameters():
            for pattern in self.always_frozen:
                if pattern.match(param):
                    data.requires_grad = False
                    print(f"freeze {param}")

            for epoch, patterns in self.schedule.items():
                for pattern in patterns:
                    if pattern.match(param):
                        data.requires_grad = False
                        print(f"freeze {param}")


    def on_epoch_start(self, state):
        if state.epoch not in self.schedule:
            return

        patterns = self.schedule[state.epoch]
        for param, data in state.model.named_parameters():
            for pattern in patterns:
                if pattern.match(param):
                    if pattern.match(param):
                        data.requires_grad = True
                        print(f"unfreeze {param}")



class LogPredsCallback(Callback):

    def __init__(self, output_key=('start_logits', 'end_logits')):
        super().__init__(order=CallbackOrder.Logging)
        self.output_key = (output_key) if isinstance(output_key, str) else output_key

    def on_epoch_start(self, state):
        self.epoch_log = {key:{} for key in self.output_key}

    def on_batch_end(self, state):
        if state.is_valid_loader:
            for key in self.output_key:
                batch_pred = state.output[key].detach().cpu().numpy()
                if state.loader_name not in self.epoch_log[key]:
                    self.epoch_log[key][state.loader_name] = []
                self.epoch_log[key][state.loader_name].append(batch_pred)

    def on_epoch_end(self, state):
        for key in self.output_key:
            self.epoch_log[key][state.loader_name].append(batch_pred)


class SheduledDropheadCallback(Callback):

    def __init__(self, cooldown_fraq=0.8, do_warmup=True):
        super().__init__(order=CallbackOrder.Internal)
        self.cooldown_fraq = cooldown_fraq
        self.do_warmup = do_warmup

    def on_stage_start(self, state):
        self.total_num_steps = len(state.loaders['train']) * state.num_epochs
        self.cooldown_num_steps = int(self.cooldown_fraq * self.total_num_steps)
        self.warmup_num_steps = self.total_num_steps - self.cooldown_num_steps
        self.max_p = state.model.roberta.encoder.layer[0].attention.self.p_drophead

    def _get_current_p(self, step):
        if step <= self.cooldown_num_steps:
            return self.max_p * (1 - step/float(self.cooldown_num_steps))
        elif not self.do_warmup:
            return 0
        else:
            cur_p = self.max_p * (step - self.cooldown_num_steps)
            cur_p /= float(self.warmup_num_steps)
            return cur_p

    def on_batch_end(self, state):
        if state.is_train_loader:
            step = state.loader_len * (state.epoch-1) + state.loader_step 
            cur_p = self._get_current_p(step)
            for bert_layer in state.model.roberta.encoder.layer:
                bert_layer.attention.self.p_drophead = cur_p


class CustomCheckpointCallback(CheckpointCallback):

    def __init__(
        self,
        save_n_best: int = 1,
        resume: str = None,
        resume_dir: str = None,
        metrics_filename: str = "_metrics.json",
        load_on_stage_end: Union[str, Dict[str, str]] = None,
        save_full = False
    ):
        super().__init__(save_n_best, resume, resume_dir, metrics_filename,
                         load_on_stage_end)
        self.save_full = save_full

    def _save_checkpoint(
        self,
        logdir: Union[str, Path],
        suffix: str,
        checkpoint: Dict,
        is_best: bool,
        is_last: bool,
    ) -> Tuple[str, str]:
        if self.save_full:
            full_checkpoint_path = utils.save_checkpoint(
                logdir=Path(f"{logdir}/checkpoints/"),
                checkpoint=checkpoint,
                suffix=f"{suffix}_full",
                is_best=is_best,
                is_last=is_last,
                special_suffix="_full",
            )
        else:
            full_checkpoint_path = None 

        exclude = ["criterion", "optimizer", "scheduler"]
        checkpoint_path = utils.save_checkpoint(
            checkpoint={
                key: value
                for key, value in checkpoint.items()
                if all(z not in key for z in exclude)
            },
            logdir=Path(f"{logdir}/checkpoints/"),
            suffix=suffix,
            is_best=is_best,
            is_last=is_last,
        )
        return (full_checkpoint_path, checkpoint_path)
