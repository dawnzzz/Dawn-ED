from torch.utils.data import Dataset
import torch


class MyDataset(Dataset):
    def __init__(self, features):
        self.nums = len(features)
        self.token_ids = [torch.tensor(example.token_ids).long() for example in features]
        self.attention_masks = [torch.tensor(example.attention_mask, dtype=torch.uint8) for example in features]
        self.token_type_ids = [torch.tensor(example.token_type_ids).long() for example in features]

        self.llf_token_ids = [torch.tensor(example.llf_token_ids).long() for example in features]
        self.llf_attention_masks = [torch.tensor(example.llf_attention_mask, dtype=torch.uint8) for example in features]
        self.llf_token_type_ids = [torch.tensor(example.llf_token_type_ids).long() for example in features]

        self.pf = [torch.LongTensor(example.pf) for example in features]

        # self.dm_token_ids = [torch.tensor(example.dm_token_ids).long() for example in features]
        # self.dm_attention_masks = [torch.tensor(example.dm_attention_mask, dtype=torch.uint8) for example in features]
        # self.dm_token_type_ids = [torch.tensor(example.dm_token_type_ids).long() for example in features]

        self.maskL = [torch.tensor(example.maskL).long() for example in features]
        self.maskR = [torch.tensor(example.maskR).long() for example in features]

        self.event_type = [torch.tensor(example.event_type, dtype=torch.float) for example in features]

        # self.bio_tags = [torch.tensor(example.bio_tags, dtype=torch.float) for example in features]

    def __len__(self):
        return self.nums

    def __getitem__(self, index):
        data = {
            'token_ids': self.token_ids[index],
            'attention_mask': self.attention_masks[index],
            'token_type_ids': self.token_type_ids[index],
            'llf_token_ids': self.llf_token_ids[index],
            'llf_attention_mask': self.llf_attention_masks[index],
            'llf_token_type_ids': self.llf_token_type_ids[index],
            'pf': self.pf[index],
            # 'dm_token_ids': self.dm_token_ids[index],
            # 'dm_attention_mask': self.dm_attention_masks[index],
            # 'dm_token_type_ids': self.dm_token_type_ids[index],
            'maskL': self.maskL[index],
            'maskR': self.maskR[index],
            'event_type': self.event_type[index],
            # 'bio_tags': self.bio_tags,
        }

        return data
