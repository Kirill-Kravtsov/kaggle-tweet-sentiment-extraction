import torch
import torch.nn.functional as F


def drophead_hook(module, input, output):
    if (not module.training) or (module.p_drophead==0):
        return output

    batch_size = output[0].shape[0]
    dist = torch.distributions.Bernoulli(torch.tensor([1-module.p_drophead]))
    mask = dist.sample((batch_size, module.num_attention_heads))
    mask = mask.to(output[0].device)
    count_ones = mask.sum(dim=1).unsqueeze(-1).unsqueeze(-1)
    mask = mask.unsqueeze(-1)

    orig_shape = output[0].shape
    self_att_res = module.transpose_for_scores(output[0])
    self_att_res = self_att_res * mask * module.num_attention_heads / count_ones
    self_att_res = self_att_res.permute(0, 2, 1, 3).view(*orig_shape)
    return (self_att_res,) + output[1:]
