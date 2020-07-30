import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data._utils.collate import default_collate

PAD_ELEMS = {
    'ids': 1,
    'mask': 0,
    'token_type_ids': 0,
    'offsets': (0, 0)
}


class DynamicPaddingCollator():

    def __init__(self, pad_keys=list(PAD_ELEMS.keys())):
        self.pad_keys = pad_keys

    def __call__(self, batch):
        sample = batch[0]
        for key in sample:
            if key in self.pad_keys:
                sample_lens = [len(s[key]) for s in batch]
                batch_len = max(sample_lens)
                is_tensor = isinstance(sample[key], torch.Tensor)
                for i, sample_len in enumerate(sample_lens):
                    pad_len = batch_len - sample_len
                    if is_tensor:
                        batch[i][key] = F.pad(batch[i][key],
                                              (0, pad_len),
                                              value=PAD_ELEMS[key])
                    else:
                        batch[i][key] += [PAD_ELEMS[key]] * pad_len
                        batch[i][key] = np.array(batch[i][key])

        return default_collate(batch)
