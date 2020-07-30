import os, sys
import random
import traceback
from functools import wraps
from multiprocessing import Process, Queue
import numpy as np
import torch
from transformers.optimization import get_linear_schedule_with_warmup
from torchcontrib.optim import SWA


def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def jaccard_func(
    start_logits,
    end_logits,
    orig_tweet,
    orig_selected,
    offsets,
    reduction=True
):
    start_pred_batch = torch.argmax(start_logits.detach(), dim=1)
    end_pred_batch = torch.argmax(end_logits.detach(), dim=1)
    jaccards = []

    if isinstance(offsets, torch.Tensor):
        offsets = offsets.cpu().numpy()
    else:
        offsets = np.array(offsets)

    for start_pred, end_pred, offset, tweet, selected in zip(
        start_pred_batch.cpu().numpy(),
        end_pred_batch.cpu().numpy(),
        offsets,
        orig_tweet,
        orig_selected
    ):
        if start_pred > end_pred:
            str_pred = ""
            jaccards.append(0)
        else:
            str_pred = tweet[offset[start_pred][0]:offset[end_pred][1]]

            str_pred = str_pred.replace('!!!!', '!') if len(str_pred.split())==1 else str_pred
            str_pred = str_pred.replace('..', '.') if len(str_pred.split())==1 else str_pred
            str_pred = str_pred.replace('...', '.') if len(str_pred.split())==1 else str_pred

            a = set(selected.lower().split())
            b = set(str_pred.lower().split())
            c = a.intersection(b)
            jaccards.append(float(len(c)) / (len(a) + len(b) - len(c)))
    if reduction:
        return np.average(jaccards)
    else:
        return np.array(jaccards)


def get_linear_schedule_with_warmup_frac(
    optimizer,
    num_training_steps,
    num_warmup_steps,
    frac_training_steps=0,
    last_epoch=-1
):
    num_warmup_steps = int(num_training_steps * frac_training_steps)
    return get_linear_schedule_with_warmup(
        optimizer,
        num_training_steps=num_training_steps,
        num_warmup_steps=num_warmup_steps,
        last_epoch=last_epoch
    )


class CustomSWA(SWA):

    def __init__(
        self,
        optimizer,
        swa_start=None,
        swa_freq=None,
        alpha=0.05
    ):
        super().__init__(optimizer, swa_start=swa_start,
                         swa_freq=swa_freq, swa_lr=None)
        self.alpha = alpha

    def update_swa_group(self, group):
        for p in group['params']:
            param_state = self.state[p]
            if 'swa_buffer' not in param_state:
                param_state['swa_buffer'] = torch.empty_like(p.data)
                param_state['swa_buffer'].copy_(p.data)
            else:
                buf = param_state['swa_buffer']
                diff = self.alpha * (p.data - buf)
                buf.add_(diff)
        group["n_avg"] += 1

    def update_model_weights(self):
        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                if 'swa_buffer' not in param_state:
                    return
                p.data.copy_(param_state['swa_buffer'])

    def update_swa_weights(self):
        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                if 'swa_buffer' not in param_state:
                    param_state['swa_buffer'] = torch.empty_like(p.data)
                param_state['swa_buffer'].copy_(p.data)


def processify(func):
    '''Decorator to run a function as a process.
    Be sure that every argument and the return value
    is *pickable*.
    The created process is joined, so the code does not
    run in parallel.
    '''

    def process_func(q, *args, **kwargs):
        try:
            ret = func(*args, **kwargs)
        except Exception as error:
            q.put((None, error))
        else:
            q.put((ret, None))

    # register original function with different name
    # in sys.modules so it is pickable
    process_func.__name__ = func.__name__ + 'processify_func'
    setattr(sys.modules[__name__], process_func.__name__, process_func)

    @wraps(func)
    def wrapper(*args, **kwargs):
        q = Queue()
        p = Process(target=process_func, args=[q] + list(args), kwargs=kwargs)
        p.start()
        ret, error = q.get()
        p.join()

        if error:
            raise error

        return ret
    return wrapper
