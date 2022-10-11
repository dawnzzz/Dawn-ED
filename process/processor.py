# -*- coding:utf-8 -*-
"""
对原始数据进行预处理
"""
from tqdm import tqdm
from typing import List
from transformers import BertTokenizer
from utils import commonUtils
import jieba
import os
import random


class InputExample:
    def __init__(self, set_type, text, tokens, triggerL, triggerR, event_type=None, labels=None):
        self.set_type = set_type  # 数据类型：test/train/dev
        self.text = text  # 文本
        self.event_type = event_type  # 事件类型
        self.tokens = tokens
        self.triggerL = triggerL
        self.triggerR = triggerR
        self.labels = labels

        # if labels is None:
        #     self.tokens = jieba.lcut(self.text)
        # else:
        #     textL = text[:labels[2]]    # labels[2] = trigger_start_index
        #     textR = text[labels[3]:]    # labels[3] = trigger_end_index
        #     tokensL = jieba.lcut(textL)
        #     tokensR = jieba.lcut(textR)
        #     self.tokens = tokensL + [labels[1]] + tokensR    # labels[1] = trigger_word
        #     self.triggerL = len(tokensL)
        #     self.triggerR = self.triggerL+1


class InputFeature:
    def __init__(self, token_ids, attention_mask, token_type_ids, maskL, maskR, llf_token_ids, llf_attention_mask, llf_token_type_ids, pf, event_type=None):
        self.token_ids = token_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids

        self.llf_token_ids = llf_token_ids
        self.llf_attention_mask = llf_attention_mask
        self.llf_token_type_ids = llf_token_type_ids

        self.pf = pf

        # self.dm_token_ids = dm_token_ids
        # self.dm_attention_mask = dm_attention_mask
        # self.dm_token_type_ids = dm_token_type_ids
        self.maskL = maskL
        self.maskR = maskR

        self.event_type = event_type
        # self.bio_tags = bio_tags


class Processor:

    def __init__(self, args):
        self.max_length = args.max_seq_len
        self.bert_dir = args.bert_dir
        self.final_data_path = os.path.join(args.data_dir, 'final_data/')
        self.raw_data_path = os.path.join(args.data_dir, 'raw_data/')
        self.label2id = {}
        self.id2label = {}

        self.llf_num = args.llf_num

        # 如果没有 labels.txt 文件， 则生成一个
        # labels.txt文件中记录删了所有的时间类型
        if not os.path.exists(os.path.join(self.final_data_path, 'labels.txt')):
            labels = []
            with open(os.path.join(self.raw_data_path, 'event_schema.json'), 'r', encoding='utf-8') as fp:
                event_schema_data = fp.read().strip()
                for line in event_schema_data.split('\n'):
                    line = eval(line)
                    labels.append(line['event_type'])

            with open(os.path.join(self.final_data_path, 'labels.txt'), 'w', encoding='utf-8') as fp:
                for label in labels:
                    fp.write(label + '\n')

        # 得到label2id和id2label
        with open(os.path.join(self.final_data_path, 'labels.txt', ), 'r', encoding='utf-8') as fp:
            labels = fp.read().strip().split('\n')
        for i, j in enumerate(labels):
            self.label2id[j] = i
            self.id2label[i] = j

    @staticmethod
    def read_json(file_path):
        with open(file_path, encoding='utf-8') as f:
            raw_examples = f.readlines()
        return raw_examples

    def get_examples(self, raw_examples, set_type):
        examples = []
        for raw_example in raw_examples:
            raw_example = eval(raw_example)
            text = raw_example['text']
            try:
                event_list = raw_example['event_list']
            except KeyError:
                # 测试集
                tokens = jieba.lcut(text)
                # TODO 补充测试数据的example生成
                for candidate in tokens:
                    pass
            else:
                # 训练集 / 验证集
                last_triggerL = 0
                for event in event_list:
                    labels = []
                    event_type = event['event_type']  # 事件类型
                    trigger_word = event['trigger']  # 触发词类型
                    # 触发词的index [trigger_start_index, trigger_end_index)
                    trigger_start_index = event['trigger_start_index']  # 触发词开始的index
                    trigger_end_index = trigger_start_index + len(trigger_word)  # 触发词的最后一个index+1

                    textL = text[:trigger_start_index]  # labels[2] = trigger_start_index
                    textR = text[trigger_end_index:]  # labels[3] = trigger_end_index
                    tokensL = jieba.lcut(textL)
                    tokensR = jieba.lcut(textR)
                    tokens = tokensL + [trigger_word] + tokensR  # labels[1] = trigger_word
                    triggerL = len(tokensL)
                    triggerR = triggerL + 1

                    labels.extend([event_type, trigger_word, trigger_start_index, trigger_end_index])
                    examples.append(InputExample(set_type=set_type,
                                                 text=text,
                                                 tokens=tokens,
                                                 triggerR=triggerR,
                                                 triggerL=triggerL,
                                                 event_type=event_type,
                                                 labels=labels,
                                                 ))

                    # 每加入一个事件，随机加入一个负样本
                    if len(tokensL[last_triggerL:]) <= 0:
                        continue
                    trigger_word = random.sample(tokensL[last_triggerL:], 1)[0]
                    last_triggerL = triggerL

                    event_type = "none"
                    trigger_start_index = text.find(trigger_word)
                    if trigger_start_index == -1:
                        continue
                    trigger_end_index = trigger_start_index + len(trigger_word)
                    textL = text[:trigger_start_index]  # labels[2] = trigger_start_index
                    textR = text[trigger_end_index:]  # labels[3] = trigger_end_index
                    tokensL = jieba.lcut(textL)
                    tokensR = jieba.lcut(textR)
                    tokens = tokensL + [trigger_word] + tokensR  # labels[1] = trigger_word
                    triggerL = len(tokensL)
                    triggerR = triggerL + 1

                    labels = []
                    labels.extend([event_type, trigger_word, trigger_start_index, trigger_end_index])
                    examples.append(InputExample(set_type=set_type,
                                                 text=text,
                                                 tokens=tokens,
                                                 triggerR=triggerR,
                                                 triggerL=triggerL,
                                                 event_type=event_type,
                                                 labels=labels,
                                                 ))

        return examples

    def get_train_features(self):

        # 如果不存在train.pkl，则根据examples生成一个
        # 存在则直接读取
        if not os.path.exists(os.path.join(self.final_data_path, 'train.pkl')):
            raw_examples = self.read_json(os.path.join(self.raw_data_path, "train.json"))
            examples = self.get_examples(raw_examples, "train")
            features = convert_examples_to_features(examples, self.max_length, self.bert_dir, self.label2id,
                                                    self.llf_num)
            commonUtils.save_pkl(self.final_data_path, features, "train")
        else:
            features = commonUtils.read_pkl(self.final_data_path, "train")

        return features

    def get_dev_features(self):

        # 如果不存在dev.pkl，则根据examples生成一个
        # 存在则直接读取
        if not os.path.exists(os.path.join(self.final_data_path, 'dev.pkl')):
            raw_examples = self.read_json(os.path.join(self.raw_data_path, "dev.json"))
            examples = self.get_examples(raw_examples, "dev")
            features = convert_examples_to_features(examples, self.max_length, self.bert_dir, self.label2id,
                                                    self.llf_num)
            commonUtils.save_pkl(self.final_data_path, features, "dev")
        else:
            features = commonUtils.read_pkl(self.final_data_path, "dev")

        return features


def convert_examples_to_features(
        examples: List[InputExample],
        max_length,
        bert_dir,
        label2id,
        llf_num
) -> List[InputFeature]:
    features = []
    tokenizer = BertTokenizer.from_pretrained(bert_dir)
    for example in examples:
        # 获取pf
        pf = []
        for i in range(example.triggerR-example.triggerL+2):    # 加2是因为两个 [unused1]
            if i >= example.triggerR-example.triggerL:
                pf.append(0)
                continue
            pf.extend([0]*len(example.tokens[example.triggerL+i]))

        for i in range(example.triggerL+1):     # 加一是因为 [CLS]
            if i >= example.triggerL:
                pf = [i+1] + pf
                continue
            pf = ([i+1]*len(example.tokens[i])) + pf

        for i in range(max_length-example.triggerR+1):     # 加一是因为 [SEP]
            if len(pf) >= max_length:
                break
            if i >= len(example.tokens)-example.triggerR:
                pf.append(i+1)
                continue
            else:
                pf.extend([i+1]*len(example.tokens[i+example.triggerR]))
        while len(pf) > max_length:
            pf.pop(-1)
        assert len(pf) == max_length

        # 获取llf相关信息
        llf = tokenizer.tokenize("".join(example.tokens[example.triggerL: example.triggerR]))
        textL = tokenizer.tokenize("".join(example.tokens[:example.triggerL]))
        textR = tokenizer.tokenize("".join(example.tokens[example.triggerR:]))
        if len(llf) + len(textL) + len(textR) >= llf_num:
            while len(llf) > llf_num:
                llf.pop()
                if len(llf) > llf_num:
                    llf.pop(len(llf) - 1)
                else:
                    break

            while len(llf) < llf_num:
                if len(textL) > 0:
                    llf = [textL.pop(len(textL) - 1)] + llf
                if len(llf) == llf_num:
                    break
                if len(textR) > 0:
                    llf.append(textR.pop(0))
        llf_inputs = tokenizer.encode_plus(llf,
                                           add_special_tokens=True,
                                           max_length=llf_num,
                                           padding="max_length",
                                           truncation=True,
                                           return_token_type_ids=True,
                                           return_overflowing_tokens=True)
        llf_input_ids, llf_token_type_ids, llf_attention_mask = llf_inputs["input_ids"], llf_inputs["token_type_ids"], llf_inputs["attention_mask"]

        # textL 是触发词左侧的分词列表
        # e.g. 雀巢裁员4000人：时代抛弃你时，连招呼都不会打！ 触发词为 “裁员”
        # textL = ['雀', '巢']
        textL = tokenizer.tokenize("".join(example.tokens[:example.triggerL]))

        # textR 是触发词以及触发词右侧的分词列表 textR = ['[unused1]', '裁', '员', '[unused1]', '4000', '人', '：', '时', '代', '抛', '弃',
        # '你', '时', '，', '连', '招', '呼', '都', '不', '会', '打', '！']
        textR = ['[unused1]']
        textR += tokenizer.tokenize("".join(example.tokens[example.triggerL:example.triggerR]))
        textR += ['[unused1]']
        textR += tokenizer.tokenize("".join(example.tokens[example.triggerR:]))

        # 前面+1 是因为最前面会加上 [CLS]
        # 后面+1 是因为最后的 [SEP]
        maskL = [1.0 for i in range(0, len(textL) + 1)] + [0.0 for i in range(0, len(textR) + 1)]
        maskR = [0.0 for i in range(0, len(textL) + 1)] + [1.0 for i in range(0, len(textR) + 1)]

        if len(maskL) > max_length:
            maskL = maskL[:max_length]
        if len(maskR) > max_length:
            maskR = maskR[:max_length]

        inputs = tokenizer.encode_plus(textL + textR,
                                       add_special_tokens=True,
                                       max_length=max_length,
                                       padding="max_length",
                                       truncation=True,
                                       return_token_type_ids=True,
                                       return_overflowing_tokens=True)

        input_ids, token_type_ids, attention_mask = inputs["input_ids"], inputs["token_type_ids"], inputs[
            "attention_mask"]

        # 在右边填充maskL和maskR
        padding_length = max_length - len(maskL)
        maskL = maskL + [0] * padding_length
        maskR = maskR + [0] * padding_length

        # 将event_type转为one-hot向量
        event_type = [0] * len(label2id)
        event_type[label2id[example.event_type]] = 1

        features.append(InputFeature(
            token_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            maskL=maskL,
            maskR=maskR,
            llf_token_ids=llf_input_ids,
            llf_attention_mask=llf_attention_mask,
            llf_token_type_ids=llf_token_type_ids,
            pf=pf,
            event_type=event_type)
        )

    return features
