import torch
import torch.nn.functional as F


def drophead_hook(module, input, output):
    att_scores = output[0]
    print(att_scrores.shape)
