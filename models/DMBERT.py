# -*- coding:utf-8 -*-
from torch import nn
from transformers import BertModel
import os
import torch
import torch.nn.functional as F


class DMBERT(nn.Module):
    
    def __init__(self, args, **kwargs):
        super(DMBERT, self).__init__()
        config_path = os.path.join(args.bert_dir, 'config.json')     # BERT 的配置文件路径

        # 断言预训练模型是否存在
        assert os.path.exists(args.bert_dir) and os.path.exists(config_path), 'pretrained bert file does not exist'

        self.bert = BertModel.from_pretrained(args.bert_dir, output_hidden_states=True)
        self.bert_config = self.bert.config
        self.out_dims = self.bert_config.hidden_size

        self.drop_out = nn.Dropout(args.dropout_prob)
        self.max_pooling = nn.MaxPool1d(args.max_seq_len)
        self.classifier = nn.Linear(self.out_dims*2, args.num_tags)

    def forward(self, **kwargs):

        token_ids = kwargs['token_ids']
        attention_mask = kwargs['attention_mask']
        token_type_ids = kwargs['token_type_ids']

        try:
            maskL = kwargs['maskL']
            maskR = kwargs['maskR']
        except KeyError:
            maskL = None
            maskR = None

        try:
            label = kwargs['event_type']
        except KeyError:
            label = None

        batch_size = token_ids.size(0)

        # 送入BERT
        outputs = self.bert(
            token_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        # 动态池化
        conv = outputs[0].permute(2, 0, 1)  # bs * seq * hidden --> hidden * bs * seq
        # maskL.shape: bs * seq
        L = (conv * maskL).transpose(0, 1)  # bs * hidden * seq
        R = (conv * maskR).transpose(0, 1)  # bs * hidden * seq

        L = L + torch.ones_like(L)  # add one to avoid overflowing
        R = R + torch.ones_like(R)

        pooledL = self.max_pooling(L).contiguous().view(batch_size, self.out_dims)   # bs * hidden
        pooledR = self.max_pooling(R).contiguous().view(batch_size, self.out_dims)   # bs * hidden

        pooled = torch.cat([pooledL, pooledR], 1)   # bs * (2*hidden)
        pooled = pooled - torch.ones_like(pooled)

        # 分类
        pooled = self.drop_out(pooled)
        logits = self.classifier(pooled)    # bs * tag_num

        if label is not None:
            return label, logits
        return F.softmax(logits, dim=1)
