import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.nn import functional as F

import pytorch_lightning as pl
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import seed_everything, LightningModule, Trainer

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from pathlib import Path
import numpy as np
import pandas as pd
import os


class ParkinsonDataset(Dataset):   
    def __init__(self, df, features, max_batch_len=4192, seq_len=128):
        self.features = features
        self.seq_len = seq_len
        self.max_batch_len = max_batch_len
        self.length = int(df["user_id"].nunique())
        self.batches = int(self.max_batch_len / self.seq_len)
        self.df = df    
        
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        user_X = self.df[self.df['user_id'] == index]
        
        y = np.repeat(np.array(user_X["labels"])[0], self.batches)
        y = torch.reshape(torch.Tensor(y), (self.batches, 1))
        
        X = torch.Tensor(np.array(user_X[self.features])).float()
        x_pad_length = self.max_batch_len - X.shape[0]
        
        padded_X = F.pad(input=X, pad=(0, 0, 0, x_pad_length), mode='constant', value=0)
        batched_X = torch.reshape(padded_X, (self.batches, self.seq_len, len(self.features)))
        batched_X = torch.Tensor(np.array(batched_X)).float()
        
        return (batched_X, y)
    
    
class ParkinsonDataModuleLSTM(pl.LightningDataModule):    
    def __init__(self, seq_len = 128, max_batch_len = 4096, num_workers=0):
        super().__init__()
        self.seq_len = seq_len
        self.max_batch_len = max_batch_len
        self.num_workers = num_workers
        self.train_df = None
        self.test_df = None
        self.val_df = None
        self.root_dir="./data/lstm"
        self.csv_file_train ="train.csv"
        self.csv_file_test ="test.csv"
        self.features = ['left_acc_x', 'left_acc_y', 'left_acc_z', 'left_gyr_x', 'left_gyr_y',
       'left_gyr_z', 'right_acc_x', 'right_acc_y', 'right_acc_z',
       'right_gyr_x', 'right_gyr_y', 'right_gyr_z']

    def prepare_data(self):
        self.df = pd.read_csv(os.path.join(self.root_dir, self.csv_file_train))
        self.test_df = pd.read_csv(os.path.join(self.root_dir, self.csv_file_test))
        
        # Create Validation Set
        split_val = self.df.groupby(["user_id", "labels"]).count().reset_index()[["user_id", "labels"]]
        self.train_idx, self.val_idx = train_test_split(split_val, stratify=split_val["labels"], test_size=0.2)

    def setup(self, stage=None):
        # Datasets
        self.train_df = self.df[self.df["user_id"].isin(self.train_idx["user_id"])].copy()        
        self.val_df = self.df[self.df["user_id"].isin(self.val_idx["user_id"])].copy()
        
        #New User ID
        flatten = lambda t: [int(item) for sublist in t for item in sublist]
        train_user_id = [np.ones(int(self.train_df.groupby(['user_id']).count().iloc[i, 0])) * i for i in range(self.train_df['user_id'].nunique())]
        val_user_id = [np.ones(int(self.val_df.groupby(['user_id']).count().iloc[i, 0])) * i for i in range(self.val_df['user_id'].nunique())]
                
        # Change original user_id with new ones starting by [0, ... , n]
        self.train_df["user_id"] = np.array(flatten(train_user_id))
        self.val_df["user_id"] = np.array(flatten(val_user_id))
        
        # Scaling
        preprocessing = StandardScaler()
        preprocessing.fit(self.train_df[self.features])

        if stage == 'fit' or stage is None:
            X_train_scaled = preprocessing.transform(self.train_df[self.features])
            self.train_df[self.features] = X_train_scaled
            
            X_val_scaled = preprocessing.transform(self.val_df[self.features])
            self.val_df[self.features] = X_val_scaled

        if stage == 'test' or stage is None:
            X_test_scaled = preprocessing.transform(self.test_df[self.features])
            self.test_df[self.features] = X_test_scaled

    def train_dataloader(self):
        train_dataset = ParkinsonDataset(self.train_df, 
                                         features = self.features,
                                         max_batch_len = self.max_batch_len, 
                                         seq_len = self.seq_len)
        
        return DataLoader(train_dataset, shuffle = False, num_workers = self.num_workers)

    def val_dataloader(self):
        val_dataset = ParkinsonDataset(self.val_df, 
                                       features = self.features,
                                       max_batch_len = self.max_batch_len, 
                                       seq_len = self.seq_len)
        
        return DataLoader(val_dataset, shuffle = False, num_workers = self.num_workers)

    def test_dataloader(self):
        test_dataset = ParkinsonDataset(self.test_df, 
                                        features = self.features,
                                        max_batch_len = self.max_batch_len, 
                                        seq_len = self.seq_len)
        
        return DataLoader(test_dataset, shuffle = False, num_workers = self.num_workers)