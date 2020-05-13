import os, sys
import random
import traceback
from functools import wraps
from multiprocessing import Process, Queue
import numpy as np
import torch
from transformers.optimization import get_linear_schedule_with_warmup


def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def jaccard_func(
    start_positions,
    end_positions,
    start_logits, 
    end_logits,
    orig_tweet,
    orig_selected,
    offsets
):
    start_pred_batch = torch.argmax(start_logits, dim=1)
    end_pred_batch = torch.argmax(end_logits, dim=1)
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
            a = set(selected.lower().split()) 
            b = set(str_pred.lower().split())
            c = a.intersection(b)
            jaccards.append(float(len(c)) / (len(a) + len(b) - len(c)))
        #if random.random() > 0.99:
        #    print()
        #    print(str_pred)
        #    print(selected)
        #    print(jaccards[-1])
    return np.average(jaccards)


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



def processify(func):
    '''Decorator to run a function as a process.
    Be sure that every argument and the return value
    is *pickable*.
    The created process is joined, so the code does not
    run in parallel.
    '''

    def process_func(q, *args, **kwargs):
        #try:
        #    ret = func(*args, **kwargs)
        ret = func(*args, **kwargs)
        #except Exception:
        #    ex_type, ex_value, tb = sys.exc_info()
        #    error = ex_type, ex_value, ''.join(traceback.format_tb(tb))
        #    ret = None
        #else:
        #    error = None

        #q.put((ret, error))
        q.put(ret)

    # register original function with different name
    # in sys.modules so it is pickable
    process_func.__name__ = func.__name__ + 'processify_func'
    setattr(sys.modules[__name__], process_func.__name__, process_func)

    @wraps(func)
    def wrapper(*args, **kwargs):
        q = Queue()
        p = Process(target=process_func, args=[q] + list(args), kwargs=kwargs)
        p.start()
        ret = q.get()
        p.join()

        #if error:
            #ex_type, ex_value, tb_str = error
            #message = '%s (in subprocess)\n%s' % (ex_value.message, tb_str)
            #print(error)
            #raise error

        return ret
    return wrapper