from torch import nn
import os
import torch
from transformers import BertModel
import torch.nn.functional as F


class DMCNN(nn.Module):

    def __init__(self, args, **kwargs):
        super(DMCNN, self).__init__()

        config_path = os.path.join(args.bert_dir, 'config.json')  # BERT 的配置文件路径

        # 断言预训练模型是否存在
        assert os.path.exists(args.bert_dir) and os.path.exists(config_path), 'pretrained bert file does not exist'

        self.bert = BertModel.from_pretrained(args.bert_dir, output_hidden_states=True)  # BERT作为预训练模型
        self.bert_config = self.bert.config
        self.bert_out_dims = self.bert_config.hidden_size

        self.pf_embedding = nn.Embedding(num_embeddings=args.max_seq_len+1, embedding_dim=args.pf_dim, padding_idx=-1)  # 位置向量

        self.cnn = _CNN(self.bert_out_dims, args)  # CNN

        self.dynamic_pooling = _DynamicPooling(args.max_seq_len)  # 动态池化层

        self.drop_out = nn.Dropout(args.dropout_prob)

        self.classifier = nn.Linear(2 * args.hidden_size + args.llf_num * self.bert_out_dims, args.num_tags)  # 分类器

    def forward(self, **kwargs):
        token_ids = kwargs['token_ids']
        attention_mask = kwargs['attention_mask']
        token_type_ids = kwargs['token_type_ids']

        try:
            pf = kwargs['pf']
        except KeyError:
            pf = None

        try:
            llf_token_ids = kwargs['llf_token_ids']
            llf_attention_mask = kwargs['llf_attention_mask']
            llf_token_type_ids = kwargs['llf_token_type_ids']
        except KeyError:
            llf_token_ids = None
            llf_attention_mask = None
            llf_token_type_ids = None

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

        # 词嵌入
        emb = self.bert(
            token_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        emb = emb[0]  # [bs, seq, bert_out_dims]
        # llf嵌入
        llf_emb = self.bert(
            llf_token_ids,
            attention_mask=llf_attention_mask,
            token_type_ids=llf_token_type_ids,
        )
        llf_emb = llf_emb[0]
        batch_size = token_ids.size(0)
        llf_emb = llf_emb.view(batch_size, -1)

        # position向量
        pf_emb = self.pf_embedding(pf)  # [bs, seq, pf_dim]

        # emb 和 pf_emb 连接起来，送入CNN中
        cnn_input = torch.cat((emb, pf_emb), dim=-1)  # [bs, seq, bert_out_dims+pf_dim]
        conv = self.cnn(cnn_input)  # [bs, hidden_size, seq]

        # 动态池化层
        pooled = self.dynamic_pooling(conv, maskL, maskR)  # bs * (2*hidden_size)
        pooled = self.drop_out(pooled)

        # 将动态池化后的向量和llf_dem连接后再送入分类器
        classifier_input = torch.cat((pooled, llf_emb), dim=-1)     # [bs, 2*hidden_size+llf_num*bert_out_dims]
        predication = self.classifier(classifier_input)

        if label is not None:
            return label, predication
        return F.softmax(predication, dim=1)


class _CNN(nn.Module):

    def __init__(self, bert_out_dims, args):
        super(_CNN, self).__init__()
        in_channels = bert_out_dims + args.pf_dim
        out_channels = args.hidden_size
        kernel_size = args.kernel_size
        padding_size = (kernel_size - 1) >> 1
        self.cnn = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                             padding=padding_size)
        self.activation = nn.ReLU()

    def forward(self, inputs):
        inputs = torch.permute(inputs, (0, 2, 1))  # [bs, seq, e_pf] -> [bs, e_pf, seq]
        prediction = self.cnn(inputs)  # [bs, hidden_size, seq]
        prediction = self.activation(prediction)  # [bs, hidden_size, seq]

        return prediction


class _DynamicPooling(nn.Module):
    def __init__(self, max_seq_len):
        super(_DynamicPooling, self).__init__()
        self.max_pooling = nn.MaxPool1d(max_seq_len)

    def forward(self, conv, maskL, maskR):
        batch_size = conv.size(0)
        # conv.shape: [bs, hidden_size, seq]
        # maskL.shape: bs * seq
        conv = conv.permute(1, 0, 2)  # [hidden_size, bs, seq]
        L = (conv * maskL).transpose(0, 1)  # bs * hidden_size * seq
        R = (conv * maskR).transpose(0, 1)  # bs * hidden_size * seq

        L = L + torch.ones_like(L)  # add one to avoid overflowing
        R = R + torch.ones_like(R)

        pooledL = self.max_pooling(L).contiguous().view(batch_size, -1)  # bs * hidden_size
        pooledR = self.max_pooling(R).contiguous().view(batch_size, -1)  # bs * hidden_size

        pooled = torch.cat([pooledL, pooledR], 1)  # bs * (2*hidden_size)
        pooled = pooled - torch.ones_like(pooled)

        return pooled
