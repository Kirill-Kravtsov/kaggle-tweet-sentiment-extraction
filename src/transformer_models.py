import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (BertPreTrainedModel, RobertaModel,
                          RobertaConfig, ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP)
from transformers import (GPT2PreTrainedModel, GPT2Model,
                          GPT2Config)
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
        multi_sample_dropout=False,
        dense_output=False
    ):
        assert layers_agg in ["concat", "sum"]
        config.attention_probs_dropout_prob = dropout
        super().__init__(config)
        self.layers_agg = layers_agg
        self.num_take_layers = int(num_take_layers)  # int because of hyperopt
        self.multi_sample_dropout = multi_sample_dropout
        self.dense_output = dense_output
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

        if dense_output:
            self.l0 = nn.Linear(lin_input_size, 1)
        else:
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

        if self.dense_output:
            return logits.squeeze(-1)
        else:
            start_logits, end_logits = logits.split(1, dim=-1)
            start_logits = start_logits.squeeze(-1)
            end_logits = end_logits.squeeze(-1)
            return start_logits, end_logits


class RobertaCNNQA(BertPreTrainedModel):
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(
        self,
        config,
        dropout=0.1,
        pre_head_dropout=0.1
    ):
        config.attention_probs_dropout_prob = dropout
        super().__init__(config)
        config.output_hidden_states = True

        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(pre_head_dropout)
        
        self.conv1_s = nn.Conv2d(1, 32, (17, config.hidden_size), padding=(8, 0))
        self.conv2_s = nn.Conv2d(32, 1, (9, 1), padding=(4, 0))
        self.conv1_e = nn.Conv2d(1, 32, (17, config.hidden_size), padding=(8, 0))
        self.conv2_e = nn.Conv2d(32, 1, (9, 1), padding=(4, 0))
        self.init_weights()

    def forward(self, ids, mask, token_type_ids):
        full_len = ids.shape[1]
        max_len = max(torch.sum((torch.sum((mask != 0), dim=0) > 0)).item(), 1)

        _, _, out = self.roberta(
            ids[:, :max_len],
            attention_mask=mask[:, :max_len],
            token_type_ids=token_type_ids[:, :max_len]
        )

        x = out[-1].unsqueeze(1)
        start_logits = F.leaky_relu(self.conv1_s(x))
        start_logits = self.conv2_s(start_logits).squeeze(1).squeeze(-1)
        end_logits = F.leaky_relu(self.conv1_e(x))
        end_logits = self.conv2_e(end_logits).squeeze(1).squeeze(-1)

        pad_len = full_len - max_len
        start_logits = F.pad(start_logits, (0, pad_len), value=0)
        end_logits = F.pad(end_logits, (0, pad_len), value=0)
        return start_logits, end_logits

class GPT2QA(GPT2PreTrainedModel):
    base_model_prefix = "gpt2"

    def __init__(
        self,
        config,
        dropout=0.1,
        pre_head_dropout=0.1,
        p_drophead=None,
        num_take_layers=2,
        freeze_embeds=False,
        layers_agg="concat",
        multi_sample_dropout=False,
        dense_output=False
    ):
        assert layers_agg in ["concat", "sum"]
        config.attn_pdrop = dropout
        super().__init__(config)
        self.layers_agg = layers_agg
        self.num_take_layers = int(num_take_layers)  # int because of hyperopt
        self.multi_sample_dropout = multi_sample_dropout
        self.dense_output = dense_output
        config.output_hidden_states = True

        self.gpt2 = GPT2Model(config)
        self.dropout = nn.Dropout(pre_head_dropout)
        if multi_sample_dropout:
            self.final_dropout = nn.Dropout(0.5)

        lin_input_size = config.n_embd
        if layers_agg == "concat":
            lin_input_size *= self.num_take_layers
        else:
            self.hid_att = nn.Linear(config.n_embd, 1)

        if dense_output:
            self.l0 = nn.Linear(lin_input_size, 1)
        else:
            self.l0 = nn.Linear(lin_input_size, 2)
        self.init_weights()
        torch.nn.init.normal_(self.l0.weight, std=0.02)

        #if p_drophead:
        #    for bert_layer in self.roberta.encoder.layer:
        #        bert_layer.attention.self.p_drophead = p_drophead
        #        bert_layer.attention.self.register_forward_hook(drophead_hook)

        if freeze_embeds:
            for name, param in self.base_model.named_parameters():
                if "embedding" in name:
                    param.requires_grad = False
        
    def forward(self, ids, mask, token_type_ids):
        full_len = ids.shape[1]
        max_len = max(torch.sum((torch.sum((mask != 0), dim=0) > 0)).item(), 1)

        _, _, out = self.gpt2(
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

        if self.dense_output:
            return logits.squeeze(-1)
        else:
            start_logits, end_logits = logits.split(1, dim=-1)
            start_logits = start_logits.squeeze(-1)
            end_logits = end_logits.squeeze(-1)
            return start_logits, end_logits
