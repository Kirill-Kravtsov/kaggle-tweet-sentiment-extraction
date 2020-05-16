import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (BertPreTrainedModel, RobertaModel,
                          RobertaConfig, ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP)
from hooks import drophead_hook


class RobertaQA(BertPreTrainedModel):
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config, dropout=0.1, p_drophead=None, num_take_layers=2):
        config.attention_probs_dropout_prob = dropout
        super().__init__(config)
        self.num_take_layers = num_take_layers
        config.output_hidden_states = True
        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(0.1)
        self.l0 = nn.Linear(config.hidden_size * num_take_layers, 2)
        self.init_weights()
        torch.nn.init.normal_(self.l0.weight, std=0.02)

        if p_drophead:
            for bert_layer in self.roberta.encoder.layer:
                bert_layer.attention.self.p_drophead = p_drophead
                bert_layer.attention.self.register_forward_hook(drophead_hook)
        
    def forward(self, ids, mask, token_type_ids):
        full_len = ids.shape[1]
        max_len = max(torch.sum((torch.sum((mask != 0), dim=0) > 0)).item(), 1)

        _, _, out = self.roberta(
            ids[:, :max_len],
            attention_mask=mask[:, :max_len],
            token_type_ids=token_type_ids[:, :max_len]
        )

        #out = torch.cat((out[-1], out[-2]), dim=-1)
        out = torch.cat([out[-(i+1)] for i in range(self.num_take_layers)], dim=-1)
        out = self.dropout(out)
        logits = self.l0(out)
        
        pad_len = full_len - max_len
        logits = F.pad(logits, (0, 0, 0, pad_len), value=0)

        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        return start_logits, end_logits
