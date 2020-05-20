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

    def __init__(
        self,
        config,
        dropout=0.1,
        pre_head_dropout=0.1,
        p_drophead=None,
        num_take_layers=2,
        freeze_embeds=False,
        layers_agg="concat",
        multi_sample_dropout=False
    ):
        assert layers_agg in ["concat", "sum"]
        config.attention_probs_dropout_prob = dropout
        super().__init__(config)
        self.layers_agg = layers_agg
        self.num_take_layers = int(num_take_layers)  # int because of hyperopt
        self.multi_sample_dropout = multi_sample_dropout
        config.output_hidden_states = True

        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(pre_head_dropout)
        if multi_sample_dropout:
            self.final_dropout = nn.Dropout(0.5)

        lin_input_size = config.hidden_size
        if layers_agg == "concat":
            lin_input_size *= self.num_take_layers
        else:
            self.hid_att = nn.Linear(config.hidden_size, 1)

        self.l0 = nn.Linear(lin_input_size, 2)
        self.init_weights()
        torch.nn.init.normal_(self.l0.weight, std=0.02)

        if p_drophead:
            for bert_layer in self.roberta.encoder.layer:
                bert_layer.attention.self.p_drophead = p_drophead
                bert_layer.attention.self.register_forward_hook(drophead_hook)

        if freeze_embeds:
            for name, param in self.base_model.named_parameters():
                if "embedding" in name:
                    param.requires_grad = False
        
    def forward(self, ids, mask, token_type_ids):
        full_len = ids.shape[1]
        max_len = max(torch.sum((torch.sum((mask != 0), dim=0) > 0)).item(), 1)

        _, _, out = self.roberta(
            ids[:, :max_len],
            attention_mask=mask[:, :max_len],
            token_type_ids=token_type_ids[:, :max_len]
        )

        #out = torch.cat((out[-1], out[-2]), dim=-1)
        out = [out[-(i+1)] for i in range(self.num_take_layers)]
        if self.layers_agg == "concat":
            out = torch.cat(out, dim=-1)
        else:
            out = torch.stack(out, dim=2)
            scores = F.softmax(self.hid_att(out).squeeze(-1), dim=2)
            out = (out * scores.unsqueeze(-1)).sum(dim=2)


        if self.multi_sample_dropout:
            logits = torch.mean(
                torch.stack(
                    [self.l0(self.final_dropout(out)) for _ in range(5)],
                    dim=0,
                ),
                dim=0,
            )
        else:
            out = self.dropout(out)
            logits = self.l0(out)
        
        pad_len = full_len - max_len
        logits = F.pad(logits, (0, 0, 0, pad_len), value=0)

        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        return start_logits, end_logits


class JaccardRobertaQA(BertPreTrainedModel):
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(
        self,
        config,
        dropout=0.1,
        pre_head_dropout=0.1,
        p_drophead=None,
        num_take_layers=2,
        freeze_embeds=False,
        layers_agg="concat"
    ):
        super().__init__(config, dropout, pre_head_dropout, p_drophead,
                         max_num_take_layers, freeze_embeds, layers_agg)
        self.lstm = nn.LSTM(
            input_size=4,
            hidden_dim=64,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        self.lstm_lin = nn.Linear(64*2, 1)


    def forward(self, ids, mask, token_type_ids,
                start_positions, end_positions, new_word_flags):
        start_logits, end_logits = super().forward(ids, mask, token_type_ids)

        selected_mask = torch.zeros_like(ids)
        selected_mask.scatter_(1, start_positions.view(-1,1), value=1)
        selected_mask.scatter_(1, end_positions.view(-1,1), value=1)
        selected_mask = torch.cumsum(selected_mask, dim=1)
        selected_mask[selected_mask==2] = 0
        selected_mask.scatter_(1, end_positions.view(-1,1), value=1)
        selected_mask[start_positions > end_positions] = 0

        lstm_inp = torch.stack((start_logits, end_logits,
                                new_word_flags, selected_mask), dim=-1)
        lstm_out, _ = self.lstm(lstm_inp)
        jaccard_pred = torch.sigmoid(self.lstm_lin)
