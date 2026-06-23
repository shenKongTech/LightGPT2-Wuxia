import os
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List

class GPTDataset(Dataset):
    def __init__(self,text,tokenizer,max_len,stride):
        self.input_ids = []
        self.target_ids = []
        token_ids = tokenizer.encode(text)
        if len(token_ids) <= max_len:
            return
        for i in range(0,len(token_ids) - max_len,stride):
            inp = token_ids[i:i + max_len]
            tgt = token_ids[i + 1:i + max_len + 1]
            self.input_ids.append(torch.tensor(inp,dtype = torch.long))
            self.target_ids.append(torch.tensor(tgt,dtype = torch.long))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

def load_and_split_data(file_list,train_ratio = 0.9):
    train_all = ''
    val_all = ''
    for fname in file_list:
        if not os.path.exists(fname):
            print(f'警告：{fname}不存在，已跳过')
            continue
        with open(fname,'r',encoding='utf-8',errors = 'ignore') as f:
            text = f.read().strip()
        split_pos = int(len(text) * train_ratio)
        train_all += text[:split_pos] + '\n'
        val_all += text[split_pos:] + '\n'
    return train_all,val_all

def create_dataloader(text,tokenizer,batch_size,max_len,stride,shuffle = True,drop_last = True):
    dataset = GPTDataset(text,tokenizer,max_len,stride)
    return DataLoader(dataset,batch_size = batch_size,shuffle = shuffle,drop_last = drop_last,num_workers=0)
