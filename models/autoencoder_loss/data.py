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
    def __init__(self, df, features, max_batch_len):
        self.features = features
        self.max_batch_len = max_batch_len        
        self.df = df
        
        self.X = torch.tensor(self.df[self.features].to_numpy()).float()
        self.y = torch.tensor(self.df["labels"].to_numpy()).float()
        
    def __len__(self):
        return int(self.df['user_id'].nunique())

    def __getitem__(self, index):
        idx = self.df[self.df['user_id'] == index].index       
        y = self.y[idx][0]
        X = self.X[idx][:self.max_batch_len]
        x_pad_length = self.max_batch_len - X.shape[0]
        
        padded_X = F.pad(input=X, pad=(0, 0, 0, x_pad_length), mode='constant', value=0)
        
        return padded_X, y
    
    
class ParkinsonResNetDataModuleAE(pl.LightningDataModule):    
    def __init__(self, max_batch_len = 4096, num_workers=0, batch_size=32):
        super().__init__()
        self.max_batch_len = max_batch_len
        self.num_workers = num_workers
        self.train_df = None
        self.test_df = None
        self.val_df = None
        self.batch_size = batch_size
        self.root_dir="./data/autoencoder_loss"
        self.csv_file_train ="Autoencoder_train.csv"
        self.csv_file_val ="Autoencoder_val.csv"
        self.csv_file_test ="Autoencoder_test.csv"

        self.features = ['left_acc_x', 'left_acc_y', 'left_acc_z', 'left_gyr_x', 'left_gyr_y',
       'left_gyr_z', 'right_acc_x', 'right_acc_y', 'right_acc_z',
       'right_gyr_x', 'right_gyr_y', 'right_gyr_z']

    def prepare_data(self):
        self.train_df = pd.read_csv(os.path.join(self.root_dir, self.csv_file_train))
        self.val_df = pd.read_csv(os.path.join(self.root_dir, self.csv_file_val))
        self.test_df = pd.read_csv(os.path.join(self.root_dir, self.csv_file_test))

    def setup(self, stage=None):        
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
        train_dataset = ParkinsonDataset(self.train_df, features = self.features, max_batch_len = self.max_batch_len)        
        return DataLoader(train_dataset, shuffle = False, num_workers = self.num_workers, batch_size = self.batch_size)

    def val_dataloader(self):
        val_dataset = ParkinsonDataset(self.val_df, features = self.features, max_batch_len = self.max_batch_len)
        return DataLoader(val_dataset, shuffle = False, num_workers = self.num_workers, batch_size = self.batch_size)

    def test_dataloader(self):
        test_dataset = ParkinsonDataset(self.test_df, features = self.features, max_batch_len = self.max_batch_len)
        return DataLoader(test_dataset, shuffle = False, num_workers = self.num_workers, batch_size = self.batch_size)
    
