import logging
from transformers.modeling_bert import BertPreTrainedModel
from transformers import BertModel

import torch
from torch import nn
from .head import Head



class CustomBert(BertPreTrainedModel):
    def __init__(self, config):
        config.output_hidden_states = True
        super(CustomBert, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(p=0.2)
        self.high_dropout = nn.Dropout(p=0.5)

        n_weights = config.num_hidden_layers + 1
        weights_init = torch.zeros(n_weights).float()
        weights_init.data[:-1] = -3
        self.layer_weights = torch.nn.Parameter(weights_init)
        self.hidden_size = config.hidden_size
        self.classifier = ClassifierWithDropout(config.hidden_size, self.config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        extra_features=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
    ):

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        hidden_layers = outputs[2]
        last_hidden = outputs[0]

        cls_outputs = torch.stack(
            [self.dropout(layer[:, 0, :]) for layer in hidden_layers],
            dim=2
        )
        cls_output = (
            torch.softmax(self.layer_weights, dim=0) * cls_outputs
        ).sum(-1)

        logits = self.classifier(cls_output)

        return logits

    def update_classifier(self, num_samples, dropout_rate=0.5):
        self.classifier = MultiSampleClassifier(
            num_samples=num_samples,
            dropout_rate=dropout_rate,
            in_features=self.hidden_size,
            out_features=self.num_labels,
        )

class DoubleBerts(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.model.siamese:
            self.bert = BertModel.from_pretrained(config.model.name, output_hidden_states=True)

        else:
            self.q_bert = BertModel.from_pretrained(config.model.name, output_hidden_states=True)
            self.a_bert = BertModel.from_pretrained(config.model.name, output_hidden_states=True)
        self.siamese = config.model.siamese
        self.dropout = nn.Dropout(p=0.2)
        n_weights = 24 + 1 if 'large' in config.model.name else 12 + 1
        weights_init = torch.zeros(n_weights).float()
        weights_init.data[:-1] = -3
        self.q_layer_weights = torch.nn.Parameter(weights_init)
        self.a_layer_weights = torch.nn.Parameter(weights_init)
        self.head = Head(n_bert=768*2)

    def forward(
        self,
        q_input_ids=None,
        q_attention_mask=None,
        q_input_segments=None,
        a_input_ids=None,
        a_attention_mask=None,
        a_input_segments=None,
        extra_features=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
    ):
        if self.siamese:
            q_outputs = self.bert(
                input_ids=q_input_ids,
                attention_mask=q_attention_mask,
                token_type_ids=q_input_segments,
            )
            a_outputs = self.bert(
                input_ids=a_input_ids,
                attention_mask=a_attention_mask,
                token_type_ids=a_input_segments,
            )

        else:
            q_outputs = self.q_bert(
                input_ids=q_input_ids,
                attention_mask=q_attention_mask,
                token_type_ids=q_input_segments,
            )
            a_outputs = self.a_bert(
                input_ids=a_input_ids,
                attention_mask=a_attention_mask,
                token_type_ids=a_input_segments,
            )

        q_cls_outputs = q_outputs[2]
        a_cls_outputs = a_outputs[2]

        q_cls_outputs = torch.stack(
            [self.dropout(layer[:, 0, :]) for layer in q_cls_outputs],
            dim=2
        )
        q_cls_outputs = (
            torch.softmax(self.q_layer_weights, dim=0) * q_cls_outputs
        ).sum(-1)

        a_cls_outputs = torch.stack(
            [self.dropout(layer[:, 0, :]) for layer in a_cls_outputs],
            dim=2
        )
        a_cls_outputs = (
            torch.softmax(self.a_layer_weights, dim=0) * a_cls_outputs
        ).sum(-1)

        q_hidden_layers = q_outputs[0]
        q_attention_mask = q_attention_mask.unsqueeze(-1)
        q_avg_outputs = (q_hidden_layers * q_attention_mask).sum(dim=1) / q_attention_mask.sum(dim=1)

        a_hidden_layers = a_outputs[0]
        a_attention_mask = a_attention_mask.unsqueeze(-1)
        a_avg_outputs = (a_hidden_layers * a_attention_mask).sum(dim=1) / a_attention_mask.sum(dim=1)

        q_outputs = torch.cat([q_cls_outputs, q_avg_outputs], dim=1)
        a_outputs = torch.cat([a_cls_outputs, a_avg_outputs], dim=1)
        # q_outputs = q_avg_outputs
        # a_outputs = a_avg_outputs

        return self.head(q_outputs, a_outputs)

def get_model(config, checkpoint_path=None):
    if config.model.num_bert == 1:
        model = CustomBert.from_pretrained(config.model.name, num_labels=config.model.num_labels)
        if config.model.num_samples > 0:
            print('info: turn on multi sample dropout')
            model.update_classifier(config.model.num_samples)
    elif config.model.num_bert == 2:
        model = DoubleBerts(config)
    model.cuda()


    if checkpoint_path != None:
        state_dict = torch.load(checkpoint_path)
        print('load model from:', checkpoint_path)
        model.load_state_dict(state_dict)

    # params = list(model.named_parameters())

    # def is_backbone(n):
    #     return "bert" in n

    # if train_only_head:
    #     for n, p in params:
    #         if is_backbone(n):
    #             p.requires_grad = False

    # grouped_parameters = [
    #     {'params': [p for n, p in params if is_backbone(n)],
    #      'lr': config.optimizer.params.encoder_lr},
    #     {'params': [p for n, p in params if not is_backbone(n)],
    #      'lr': config.optimizer.params.decoder_lr}
    # ]

    # return model, grouped_parameters
    return model

