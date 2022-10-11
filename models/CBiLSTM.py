# -*- coding:utf-8 -*-
from torch import nn
from transformers import BertModel
import torch.nn.functional as F
import torch


class CBiLSTM(nn.Module):

    def __init__(self, args, **kwargs):
        super(CBiLSTM, self).__init__()
        self.bert = BertModel.from_pretrained(args.bert_dir, output_hidden_states=True)
        self.bert_config = self.bert.config
        self.bert_out_dims = self.bert_config.hidden_size

        self.bilstm = nn.LSTM(
            input_size=self.bert_out_dims,
            hidden_size=self.bert_out_dims//2,
            batch_first=True,
            num_layers=2,
            bias=True,
            bidirectional=True
        )

        self.kernel_sizes = [3, 5, 7]
        self.cnn_out_dims = [args.max_seq_len-ks+1 for ks in self.kernel_sizes]
        self.cnns = [nn.Conv1d(in_channels=self.bert_out_dims, out_channels=self.bert_out_dims, kernel_size=ks) for ks in self.kernel_sizes]

        self.dropout = nn.Dropout(args.dropout_prob)
        self.fc = nn.Linear(
            in_features=(args.max_seq_len+3)*self.bert_out_dims,
            out_features=args.num_tags,     # 用 BIO 标记
            bias=True
        )

    def forward(self, **kwargs):
        token_ids = kwargs['token_ids']
        attention_mask = kwargs['attention_mask']
        token_type_ids = kwargs['token_type_ids']

        try:
            label = kwargs['event_type']
        except KeyError:
            label = None

        # 送入BERT
        outputs = self.bert(
            token_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )   # bs * seq * bert_out_dims
        batch_size = token_ids.size(0)
        outputs = outputs[0]

        # 送入BiLSTM
        lstm_out, _ = self.bilstm(outputs)  # bs * seq * self.bert_out_dims

        # 送入CNN
        cnn_out_list = []
        i = 0
        for cnn in self.cnns:
            # outputs.permute(0, 2, 1): bs * ber_out_dims * seq
            conv_out = cnn(outputs.permute(0, 2, 1))   # bs * self.bert_out_dims * cnn_out_dims[i]
            pooled_out = F.max_pool1d(conv_out, self.cnn_out_dims[i]) # bs * self.bert_out_dims * 1
            cnn_out_list.append(pooled_out)
            i += 1
        cnn_out = torch.cat(cnn_out_list, dim=2)    # bs * self.bert_out_dims * 3
        cnn_out = cnn_out.view(batch_size, -1)  # bs * (3*self.bert_out_dims)

        # 将BiLSTM和CNN的结果拼接起来
        lstm_out = lstm_out.reshape(batch_size, -1)    # bs * (seq*self.bert_out_dims)
        cated = torch.cat([lstm_out, cnn_out], dim=1)   # bs * (3*self.bert_out_dims+seq*self.bert_out_dims)

        # 送入线性层进行分类
        out = self.dropout(cated)
        out = self.fc(out)
        # lstm_out_list = lstm_out.split(1, dim=1)
        # for each in lstm_out_list:  # 每一个 LSTM 的输出与CNN拼接后送入线性层进行分类
        #     each = each.squeeze(dim=1)  # bs * self.bert_out_dims
        #     cated = torch.cat([each, cnn_out], dim=1)   # bs * (self.bert_out_dims+3*self.bert_out_dims)
        #     out = self.dropout(cated)
        #     out = self.fc(out)      # bs * tag_num
        if label is not None:
            return label, out
        return F.softmax(out, dim=1)

