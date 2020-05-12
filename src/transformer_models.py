import torch
import torch.nn as nn
from transformers import (BertPreTrainedModel, RobertaModel,
                          RobertaConfig, ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP)


class RobertaQA(BertPreTrainedModel):
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config):
        super().__init__(config)
        config.output_hidden_states = True
        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(0.1)
        self.l0 = nn.Linear(config.hidden_size * 2, 2)
        self.init_weights()
        torch.nn.init.normal_(self.l0.weight, std=0.02)
        
    def forward(self, ids, mask, token_type_ids):
        _, _, out = self.roberta(
            ids,
            attention_mask=mask,
            token_type_ids=token_type_ids
        )

        out = torch.cat((out[-1], out[-2]), dim=-1)
        out = self.dropout(out)
        logits = self.l0(out)

        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        return start_logits, end_logits
