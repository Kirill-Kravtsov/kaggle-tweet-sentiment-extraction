from catalyst.core import Callback, MetricCallback, CallbackOrder
from utils import jaccard_func, CustomSWA


class JaccardCallback(MetricCallback):

    def __init__(
        self,
        input_key = ['start_positions', 'end_positions', 'orig_tweet', 'orig_selected', 'offsets'],
        output_key = ['start_logits', 'end_logits'],
        prefix = "jaccard",
        activation = None
    ):
        super().__init__(
            prefix=prefix,
            metric_fn=jaccard_func,
            input_key=input_key,
            output_key=output_key
            #activation=activation
        )


class SWACallback(Callback):

    def __init__(self, swa_start=10, swa_freq=1, alpha=0.05):
        super().__init__(order=CallbackOrder.Internal)
        self.swa_start = swa_start
        self.swa_freq = swa_freq
        self.alpha = alpha
        self.swapped = False

    def on_stage_start(self, state):
        state.optimizer = CustomSWA(
            optimizer=state.optimizer,
            swa_start=self.swa_start,
            swa_freq=self.swa_freq,
            alpha=self.alpha
        )

    def on_loader_end(self, state):
        if (state.is_train_loader and self.swapped):
            state.optimizer.swap_swa_sgd()
            self.swapped = False
        elif (state.is_valid_loader and not self.swapped):
            state.optimizer.swap_swa_sgd()
            self.swapped = True
