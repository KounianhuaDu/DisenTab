import torch
import os
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm, trange
import numpy as np
import pickle as pkl

class BaseDataset(Dataset):
    def __init__(self, target_data):
        with open(target_data, 'rb') as f:
            lines = pkl.load(f)
        self.user_cols = lines[0]
        self.item_cols = lines[1]
        self.context_cols = lines[2]
        self.labels = lines[3]
        self.hists = lines[4]
        self.hist_len = lines[5]
        self.len = self.user_cols.shape[0]

    def __getitem__(self, index):
        x_user = torch.LongTensor(self.user_cols[index])
        x_item = torch.LongTensor(self.item_cols[index])
        x_context = torch.LongTensor(self.context_cols[index])
        hists = self.hists[index]
        label = self.labels[index]
        hist_len = self.hist_len[index]
        return x_user, x_item, x_context, torch.LongTensor(hists), torch.tensor(hist_len), torch.tensor(label)

    def __len__(self):
        return self.len     

def load_data(dataset, batch_size, num_workers=8, path='../data',sampled=False):
    path = os.path.join(path, dataset, 'splited_data')
    
    train_data = os.path.join(path, 'full_train_data.pkl')
    valid_data = os.path.join(path, 'valid_pool.pkl')
    test_data = os.path.join(path, 'test_pool.pkl')

    train_dataset = BaseDataset(train_data)
    valid_dataset = BaseDataset(valid_data)
    test_dataset = BaseDataset(test_data)

    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader= DataLoader(
        dataset=valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader= DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, valid_loader, test_loader
