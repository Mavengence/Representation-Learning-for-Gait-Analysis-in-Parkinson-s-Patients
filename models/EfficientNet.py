import matplotlib.pyplot as plt
#%matplotlib widget
import numpy as np
import pandas as pd
from pathlib import Path

from pytorch_lightning.metrics.functional import f1_score, recall, precision, accuracy, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os
from efficientnet_pytorch import EfficientNet

p= os.path.abspath("Untitled.ipynb")
data_path = Path(p + '/../data/')
participant_labels = pd.read_csv(data_path.joinpath('Labels.csv'))
participant_labels.head()

example = next(data_path.iterdir())
sensor_data = list(example.glob('*.csv'))

participant_labels['labels'] = 0

participant_labels['Hoehn_Yahr'] = participant_labels['Hoehn_Yahr'].replace(np.nan, 0.0)
participant_labels['UPDRS_Motor_Scale'] = participant_labels['UPDRS_Motor_Scale'].replace(np.nan, 0.0)

participant_labels.loc[(participant_labels['Hoehn_Yahr'] > 0.0) & (participant_labels['UPDRS_Motor_Scale'] > 0.0), "labels"] = 1

X_train, X_test, y_train, y_test = train_test_split(participant_labels['STUD_Ganganalyse'],
                 participant_labels["labels"],
                 stratify=participant_labels[["labels", "Geschlecht"]],
                 test_size=0.2)

train_df = pd.DataFrame()

missing = 0
for i, user in enumerate(X_train):
    left_path = os.path.join(data_path, user, "sensor_data_left.csv")
    right_path = os.path.join(data_path, user, "sensor_data_right.csv")

    try:
        df_l = pd.read_csv(left_path).add_prefix("left_").drop(columns=["left_Unnamed: 0"])
        df_r = pd.read_csv(left_path).add_prefix("right_").drop(columns=["right_Unnamed: 0"])
    except:
        missing += 1
        print("Missing CSV")

    if len(df_r) != len(df_l):
        missing += 1
        print("Not the same length")

    else:
        df_user = pd.concat([df_l, df_r], axis=1)
        df_user['user_id'] = i
        # Das ist neu!
        # Check ob der Patient mehr als x daten hat (hier 4000)
        # True => cut bei x
        # False => Padding mit Nullen
        if df_user.shape[0] > 4000:
            df_user = df_user.loc[:3999, :]
        else:
            pad = pd.DataFrame(0, index=np.arange(4000 - df_user.shape[0]),
                               columns=['left_acc_x', 'left_acc_y', 'left_acc_z', 'left_gyr_x', 'left_gyr_y',
                                        'left_gyr_z', 'right_acc_x', 'right_acc_y', 'right_acc_z', 'right_gyr_x',
                                        'right_gyr_y', 'right_gyr_z'])
            pad['user_id'] = i
            df_user = df_user.append(pad, ignore_index=True)
        train_df = train_df.append(df_user, ignore_index=True)
print(f"{missing} Users")

test_df = pd.DataFrame()

missing = 0

for i, user in enumerate(X_test):
    left_path = os.path.join(data_path, user, "sensor_data_left.csv")
    right_path = os.path.join(data_path, user, "sensor_data_right.csv")

    try:
        df_l = pd.read_csv(left_path).add_prefix("left_").drop(columns=["left_Unnamed: 0"])
        df_r = pd.read_csv(left_path).add_prefix("right_").drop(columns=["right_Unnamed: 0"])
    except:
        missing += 1
        print("Missing CSV")

    if len(df_r) != len(df_l):
        missing += 1
        print("Not the same length")

    else:
        df_user = pd.concat([df_l, df_r], axis=1)
        df_user['user_id'] = i
        # Das ist neu!
        # Check ob der Patient mehr als x daten hat (hier 4000)
        # True => cut bei x
        # False => Padding mit Nullen
        if df_user.shape[0] > 4000:
            df_user = df_user.loc[:3999, :]
        else:
            pad = pd.DataFrame(0, index=np.arange(4000 - df_user.shape[0]),
                               columns=['left_acc_x', 'left_acc_y', 'left_acc_z', 'left_gyr_x', 'left_gyr_y',
                                        'left_gyr_z', 'right_acc_x', 'right_acc_y', 'right_acc_z', 'right_gyr_x',
                                        'right_gyr_y', 'right_gyr_z'])
            pad['user_id'] = i
            df_user = df_user.append(pad, ignore_index=True)
        test_df = test_df.append(df_user, ignore_index=True)

print(f"{missing} Users")

#drop user_id
print(y_train.hist())
print(y_test.hist())
print(f"Train_df: {train_df.shape}")
print(f"Test_df: {test_df.shape}")
x_train = train_df.drop(columns="user_id")
x_test = test_df.drop(columns="user_id")

"""

class customTrainData(Dataset):

    def __init__(self):
        print("SHAPE")
        print(x_train.values.shape)
        self.x = torch.tensor(x_train.values[:80000], dtype=torch.float32)
        self.y = torch.tensor(y_train.values[:20], dtype=torch.float32)
        print(self.x.size())
        self.nsamples = 20
        self.x = self.x.view(20, 1, 4000, 12)
        self.y = self.y.view(20, 1)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.nsamples


# DataSet for validation data
class customValData(Dataset):

    def __init__(self):
        self.x = torch.tensor(x_test.values, dtype=torch.float32)
        self.y = torch.tensor(y_test.values, dtype=torch.float32)
        self.nsamples = 59
        self.x = self.x.view(59, 1, 4000, 12)
        self.y = self.y.view(59, 1)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.nsamples
    
    
class ResNet(pl.LightningModule):

    def __init__(self):
        super(ResNet, self).__init__()
        #pytorch lightning object to get the accuracy
        self.acc = pl.metrics.Accuracy()
        #Old ResNet from DL course
        self.elu1 = nn.ReLU()
        self.elu2 = nn.ReLU()
        self.elu3 = nn.ReLU()
        self.elu4 = nn.ReLU()
        self.elu5 = nn.ReLU()
        self.elu6 = nn.ReLU()
        self.elu7 = nn.ReLU()
        self.elu8 = nn.ReLU()
        self.elu9 = nn.ReLU()
        self.conv1 = nn.Conv2d(1, 64, 12)
        self.maxpool1 = nn.MaxPool2d(3, 3)
        # Resblock 1
        self.conv2 = nn.Conv2d(64, 64, 3, 1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, 1, padding=1)
        self.conv1x11 = nn.Conv2d(64, 64, 1, 1)
        # Resblock 2
        self.conv4 = nn.Conv2d(64, 128, 3, 2, padding=1)
        self.conv5 = nn.Conv2d(128, 128, 3, 1, padding=1)
        self.conv1x12 = nn.Conv2d(64, 128, 1, 2)
        # Resblock 3
        self.conv6 = nn.Conv2d(128, 256, 3, 2, padding=1)
        self.conv7 = nn.Conv2d(256, 256, 3, 1, padding=1)
        self.conv1x13 = nn.Conv2d(128, 256, 1, 2)
        # Resblock 4
        self.conv8 = nn.Conv2d(256, 512, 3, 2, padding=1)
        self.conv9 = nn.Conv2d(512, 512, 3, 1, padding=1)
        self.conv1x14 = nn.Conv2d(256, 512, 1, 2)
        # END
        self.avgpool1 = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(512, 512)
        self.elu10 = nn.ELU(inplace=False)
        self.fc2 = nn.Linear(512, 512)
        self.elu11 = nn.ELU(inplace=False)
        self.fc3 = nn.Linear(512, 512)
        self.elu12 = nn.ELU(inplace=False)
        self.fc4 = nn.Linear(512, 1)
        self.sigmoid1 = nn.Sigmoid()
        """"""self.train_acc = pl.metrics.Accuracy()
        self.val_acc = pl.metrics.Accuracy()
        self.efficient_net = EfficientNet.from_name('efficientnet-b0')
        # if you have acces to internet use just \
        # use this- EfficientNet.from_pretrained('efficientnet-b3',num_classes=CLASSES)
        in_features = self.efficient_net._fc.in_features
        self.efficient_net._conv_stem = nn.Conv2d(1, 32, 12)
        self.efficient_net._dropout = nn.Dropout(p=0.96, inplace=False)
        self.efficient_net._fc = nn.Linear(in_features, 1)
        self.efficient_net._swish = nn.Identity()
        print(self.efficient_net)
        self.sig = nn.Sigmoid()""""""

    def forward(self, x):
        x = self.conv1(x)
        x = self.elu1(x)
        identity = x
        # Resblock 1
        x = self.conv2(x)
        x = self.elu2(x)
        x = self.conv3(x)
        x = self.elu3(x)
        identity = self.conv1x11(identity)
        x += identity
        identity = x
        # Resblock 2
        x = self.conv4(x)
        x = self.elu4(x)
        x = self.conv5(x)
        x = self.elu5(x)
        identity = self.conv1x12(identity)
        x += identity
        identity = x
        # Resblock 3
        x = self.conv6(x)
        x = self.elu6(x)
        x = self.conv7(x)
        x = self.elu7(x)
        identity = self.conv1x13(identity)
        x += identity
        identity = x
        # Resblock 4
        x = self.conv8(x)
        x = self.elu8(x)
        x = self.conv9(x)
        x = self.elu9(x)
        identity = self.conv1x14(identity)
        x += identity
        x = self.avgpool1(x)
        x = x.flatten(start_dim=1)
        x = self.fc1(x)
        x = self.elu10(x)
        x = self.fc2(x)
        x = self.elu11(x)
        x = self.fc3(x)
        x = self.elu12(x)
        x = self.fc4(x)
        x = self.sigmoid1(x)

        #x = self.efficient_net(x)
        #x = self.sig(x)

        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        images, labels = batch

        # Forward pass
        outputs = self(images)
        print(outputs)
        loss = F.binary_cross_entropy_with_logits(outputs, labels)

        batch_f1 = f1_score(outputs, labels)
        batch_prec = precision(outputs, labels)
        batch_rec = recall(outputs, labels)
        mat = confusion_matrix(outputs, labels, num_classes=2)
        print(labels)
        print(outputs)
        print(mat)
        return {'loss': loss, 'f1': batch_f1, 'prec': batch_prec, 'rec': batch_rec}

        #self.train_acc(outputs, labels)
        #self.log('train_acc', self.train_acc, on_step=True, on_epoch=False)
        #return {'loss': loss}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        f1 = torch.stack([x['f1'] for x in outputs]).mean() * 100
        prec = torch.stack([x['prec'] for x in outputs]).mean() * 100
        rec = torch.stack([x['rec'] for x in outputs]).mean() * 100
        logs = {'avg_train_loss': avg_loss, 'f1': f1}
        self.log('avg_train_loss', avg_loss)
        self.log('f1', f1)
        self.log('precision_train', prec)
        self.log('recall_train', rec)
        #return {'avg_train_loss': avg_loss,
                #'progress_bar': logs, 'log': logs}

    def validation_step(self, batch, batch_idx):
        data, labels = batch

        # Forward pass
        outputs = self(data)

        #loss = F.binary_cross_entropy(outputs, labels)

        acc = accuracy(outputs, labels)
        prec = precision(outputs, labels)
        rec = recall(outputs, labels)
        mat = confusion_matrix(outputs, labels, num_classes=2)

        print(labels)
        print(outputs)
        print(mat)
        return {'val_loss': acc, 'val_acc': acc, 'prec': prec, 'rec': rec}

    def train_dataloader(self):
        dataset = customTrainData()
        train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=64, shuffle=True, num_workers=8)
        return train_loader

    def val_dataloader(self):
        dataset_val = customValData()
        val_loader = torch.utils.data.DataLoader(dataset=dataset_val, batch_size=64, num_workers=8)
        return val_loader

    def validation_epoch_end(self, outputs):
        # outputs = 0
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        avg_rec = torch.stack([x['rec'] for x in outputs]).mean()
        avg_prec = torch.stack([x['prec'] for x in outputs]).mean()
        #tensorboard_logs = {'avg_val_loss': avg_loss}
        self.log('valid_loss', avg_loss)
        self.log('valid_acc', avg_acc)
        self.log('valid_rec', avg_rec)
        self.log('valid_prec', avg_prec)


trainer = pl.Trainer(max_epochs=10, fast_dev_run=False)
model = ResNet()
trainer.fit(model)
trainer.test()
"""


# DataSet for train data
class customTrainData(Dataset):

    def __init__(self):
        self.x = torch.tensor(x_train.values, dtype=torch.float32)
        self.y = torch.tensor(y_train.values, dtype=torch.float32)
        scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
        scaler.fit(np.transpose(self.x))
        self.x = np.transpose(scaler.transform(np.transpose(self.x)))
        #self.transform = transforms.Normalize(self.x.mean(), self.x.std())
        self.x = torch.tensor(self.x, dtype=torch.float32)
        self.nsamples = 233
        self.x = self.x.view(233, 1, 4000, 12)
        self.y = self.y.view(233, 1)

    def __getitem__(self, index):
        return self.x[index], self.y[index]#self.transform(self.x[index]), self.y[index]

    def __len__(self):
        return self.nsamples


# DataSet for validation data
class customValData(Dataset):

    def __init__(self):
        self.x = torch.tensor(x_test.values, dtype=torch.float32)
        self.y = torch.tensor(y_test.values, dtype=torch.float32)
        scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
        scaler.fit(np.transpose(self.x))
        self.x = np.transpose(scaler.transform(np.transpose(self.x)))
        #self.transform = transforms.Normalize(self.x.mean(), self.x.std())
        self.x = torch.tensor(self.x, dtype=torch.float32)
        self.nsamples = 59
        self.x = self.x.view(59, 1, 4000, 12)
        self.y = self.y.view(59, 1)

    def __getitem__(self, index):
        return self.x[index], self.y[index]#self.transform(self.x[index]), self.y[index]

    def __len__(self):
        return self.nsamples

#Sorry for the bad code this is just my testing stuff .....
class ResNet(pl.LightningModule):

    def __init__(self, learning_rate):
        super(ResNet, self).__init__()
        """# pytorch lightning object to get the accuracy
        self.acc = pl.metrics.Accuracy()
        self.learning_rate = learning_rate
        self.batch_size = 16
        # Old ResNet from DL course
        self.elu1 = nn.LeakyReLU()
        self.elu2 = nn.LeakyReLU()
        self.elu3 = nn.LeakyReLU()
        self.elu4 = nn.LeakyReLU()
        self.elu5 = nn.LeakyReLU()
        self.elu6 = nn.LeakyReLU()
        self.elu7 = nn.LeakyReLU()
        self.elu8 = nn.LeakyReLU()
        self.elu9 = nn.LeakyReLU()
        self.elu10 = nn.LeakyReLU()
        self.elu11 = nn.LeakyReLU()
        self.elu12 = nn.LeakyReLU()
        self.elu13 = nn.LeakyReLU()
        self.elu14 = nn.LeakyReLU()
        self.elu15 = nn.LeakyReLU()
        self.elu16 = nn.LeakyReLU()
        self.elu17 = nn.LeakyReLU()
        self.batchnorm1 = nn.BatchNorm2d(64)
        self.batchnorm2 = nn.BatchNorm2d(64)
        self.batchnorm3 = nn.BatchNorm2d(64)
        self.batchnorm4 = nn.BatchNorm2d(128)
        self.batchnorm5 = nn.BatchNorm2d(128)
        self.batchnorm6 = nn.BatchNorm2d(256)
        self.batchnorm7 = nn.BatchNorm2d(256)
        self.batchnorm8 = nn.BatchNorm2d(512)
        self.batchnorm9 = nn.BatchNorm2d(512)
        self.batchnorm10 = nn.BatchNorm2d(512)
        self.batchnorm11 = nn.BatchNorm2d(512)
        self.batchnorm12 = nn.BatchNorm2d(512)
        self.batchnorm13 = nn.BatchNorm2d(512)
        self.batchnorm14 = nn.BatchNorm2d(512)
        self.batchnorm15 = nn.BatchNorm2d(512)
        self.batchnorm16 = nn.BatchNorm2d(512)
        self.batchnorm17 = nn.BatchNorm2d(512)
        self.conv1 = nn.Conv2d(1, 64, (10, 6), 1)#increased this from 10 to 50
        self.maxpool1 = nn.MaxPool2d(3, 3)
        # Resblock 1
        self.conv2 = nn.Conv2d(64, 64, 3, 1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, 1, padding=1)
        self.conv1x11 = nn.Conv2d(64, 64, 1, 1)
        # Resblock 2
        self.conv4 = nn.Conv2d(64, 128, 3, 2, padding=1)
        self.conv5 = nn.Conv2d(128, 128, 3, 1, padding=1)
        self.conv1x12 = nn.Conv2d(64, 128, 1, 2)
        # Resblock 3
        self.conv6 = nn.Conv2d(128, 256, 3, 2, padding=1)
        self.conv7 = nn.Conv2d(256, 256, 3, 1, padding=1)
        self.conv1x13 = nn.Conv2d(128, 256, 1, 2)
        # Resblock 4
        self.conv8 = nn.Conv2d(256, 512, 3, 2, padding=1)
        self.conv9 = nn.Conv2d(512, 512, 3, 1, padding=1)
        self.conv1x14 = nn.Conv2d(256, 512, 1, 2)
        # Resblock 5
        self.conv10 = nn.Conv2d(512, 512, 3, 2, padding=1)
        self.conv11 = nn.Conv2d(512, 512, 3, 1, padding=1)
        self.conv1x15 = nn.Conv2d(512, 512, 1, 2)
        # Resblock 6
        self.conv12 = nn.Conv2d(512, 512, 3, 2, padding=1)
        self.conv13 = nn.Conv2d(512, 512, 3, 1, padding=1)
        self.conv1x16 = nn.Conv2d(512, 512, 1, 2)
        # Resblock 7
        self.conv14 = nn.Conv2d(512, 512, 3, 2, padding=1)
        self.conv15 = nn.Conv2d(512, 512, 3, 1, padding=1)
        self.conv1x17 = nn.Conv2d(512, 512, 1, 2)
        # Resblock 8
        self.conv16 = nn.Conv2d(512, 512, 3, 2, padding=1)
        self.conv17 = nn.Conv2d(512, 512, 3, 1, padding=1)
        self.conv1x18 = nn.Conv2d(512, 512, 1, 2)
        # END
        self.avgpool1 = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(512, 512)
        self.elu10 = nn.LeakyReLU()
        self.fc2 = nn.Linear(512, 512)
        self.elu11 = nn.LeakyReLU()
        self.fc3 = nn.Linear(512, 512)
        self.elu12 = nn.LeakyReLU()
        self.fc4 = nn.Linear(512, 512)
        self.elu18 = nn.LeakyReLU()
        self.fc6 = nn.Linear(512, 512)
        self.elu19 = nn.LeakyReLU()
        self.fc7 = nn.Linear(512, 512)
        self.elu13 = nn.LeakyReLU()
        self.fc5 = nn.Linear(512, 1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.1)
        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.2)
        self.dropout3 = nn.Dropout(p=0.2)
        self.dropout4 = nn.Dropout(p=0.3)
        self.dropout5 = nn.Dropout(p=0.3)"""


        """#self.acc = pl.metrics.Accuracy()
        #self.learning_rate = learning_rate
        #self.resnet = torchvision.models.resnet34(pretrained=False
                                                  )
        self.resnet.fc = nn.Linear(512, 1)
        self.conv = nn.Conv2d(1, 3, 1, 1)"""
        self.batch_size = 16
        #replace .from_name('efficientnet-b1') with .from_pretrained('efficientnet-b1') to get a pretrained version
        #BUG IF .from_name('efficientnet-b1') IS USED THE OUTPUT OF THE VALIDATION STEP WILL BE THE SAME FOR ANY INPUT
        self.efficient_net = EfficientNet.from_name('efficientnet-b1')
        in_features = self.efficient_net._fc.in_features
        print(self.efficient_net)
        self.efficient_net._conv_stem = nn.Conv2d(1, 32, 12)
        self.learning_rate = learning_rate
        #change the last fc layer to classify into 1 or 0
        self.efficient_net._fc = nn.Linear(in_features, 1)
        #use sigmoid for mapping between 0 and 1 necessary for bce loss
        self.sig = nn.Sigmoid()

    def forward(self, x):
        """x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.elu1(x)
        identity = x
        # Resblock 1
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.elu2(x)
        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = self.elu3(x)
        identity = self.conv1x11(identity)
        x += identity
        identity = x
        # Resblock 2
        x = self.conv4(x)
        x = self.batchnorm4(x)
        x = self.elu4(x)
        x = self.conv5(x)
        x = self.batchnorm5(x)
        x = self.elu5(x)
        identity = self.conv1x12(identity)
        x += identity
        identity = x
        # Resblock 3
        x = self.conv6(x)
        x = self.batchnorm6(x)
        x = self.elu6(x)
        x = self.conv7(x)
        x = self.batchnorm7(x)
        x = self.elu7(x)
        identity = self.conv1x13(identity)
        x += identity
        identity = x
        # Resblock 4
        x = self.conv8(x)
        x = self.batchnorm8(x)
        x = self.elu8(x)
        x = self.conv9(x)
        x = self.batchnorm9(x)
        x = self.elu9(x)
        identity = self.conv1x14(identity)
        x += identity
        identity = x
        # Resblock 5
        x = self.conv10(x)
        x = self.batchnorm10(x)
        x = self.elu10(x)
        x = self.conv11(x)
        x = self.batchnorm11(x)
        x = self.elu11(x)
        identity = self.conv1x15(identity)
        x += identity
        identity = x
        # Resblock 6
        x = self.conv12(x)
        x = self.batchnorm12(x)
        x = self.elu12(x)
        x = self.conv13(x)
        x = self.batchnorm13(x)
        x = self.elu13(x)
        identity = self.conv1x16(identity)
        x += identity
        # Resblock 7
        x = self.conv14(x)
        x = self.batchnorm14(x)
        x = self.elu14(x)
        x = self.conv15(x)
        x = self.batchnorm15(x)
        x = self.elu15(x)
        identity = self.conv1x17(identity)
        x += identity
        # Resblock 8
        x = self.conv16(x)
        x = self.batchnorm16(x)
        x = self.elu16(x)
        x = self.conv17(x)
        x = self.batchnorm17(x)
        x = self.elu17(x)
        identity = self.conv1x18(identity)
        x += identity
        x = self.avgpool1(x)
        x = x.flatten(start_dim=1)
        x = self.dropout(self.fc1(x))
        x = self.elu10(x)
        x = self.dropout1(self.fc2(x))
        x = self.elu11(x)
        x = self.dropout2(self.fc3(x))
        x = self.elu12(x)
        x = self.dropout3(self.fc4(x))
        x = self.elu18(x)
        x = self.dropout4(self.fc6(x))
        x = self.elu19(x)
        x = self.dropout5(self.fc7(x))
        x = self.elu13(x)
        x = self.fc5(x)
        x = self.sigmoid(x)
        #x = self.conv(x)
        #x = self.resnet(x)
        #x = self.dropout(self.fc(x))


        #x = self.relu(x)
        #x = self.dropout1(self.fc1(x))
        #x = self.relu1(x)
        #x = self.dropout2(self.fc2(x))
        #x = self.relu2(x)
        #x = self.dropout3(self.fc3(x))"""
        x = self.efficient_net(x)
        x = self.sig(x)

        return x

    def configure_optimizers(self):
        print(self.learning_rate)
        optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,self.learning_rate)
        return optimizer#], [scheduler]

    def training_step(self, batch, batch_idx):
        images, labels = batch

        # Forward pass
        outputs = self(images)

        #since the classes are imbalanced use weighting
        tensor = torch.zeros([labels.size()[0], 1], dtype=torch.float32)
        weight = 0
        counter = 0
        for i in labels:
            weight += i.item()

        for i in labels:
            if i.item() == 1:
                tensor[counter][0] = weight/labels.size()[0]
            else:
                tensor[counter][0] = (labels.size()[0] - weight) / labels.size()[0]
            counter += 1

        loss = F.binary_cross_entropy(outputs, labels, weight=tensor)
        print(labels)
        print(outputs)

        return {'loss': loss}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('avg_train_loss', avg_loss)

    def validation_step(self, batch, batch_idx):
        data, labels = batch

        # Forward pass
        outputs = self(data)

        print(outputs)
        tensor = torch.zeros([labels.size()[0], 1], dtype=torch.float32)
        weight = 0
        counter = 0
        for i in labels:
            weight += i.item()

        for i in labels:
            if i.item() == 1:
                tensor[counter][0] = weight/labels.size()[0]
            else:
                tensor[counter][0] = (labels.size()[0] - weight) / labels.size()[0]
            counter += 1
        loss = F.binary_cross_entropy(outputs, labels, weight=tensor)
        #Calculate some metrics
        acc = (labels == torch.round(outputs)).sum()
        ones = (torch.round(outputs) == 1)
        true_positives = (labels[ones] == 1).sum()
        zeros = (torch.round(outputs) == 0)
        false_positives = (labels[ones] == 0).sum()
        false_negatives = (labels[zeros] == 1).sum()
        true_negatives = (labels[zeros] == 0).sum()
        positives = false_positives + true_positives
        if positives == 0:
            precision = torch.tensor(0, dtype=torch.float32)
        else:
            precision = (true_positives / positives)

        if (true_negatives + false_negatives) == 0:
            specificity = torch.tensor(0, dtype=torch.float32)
        else:
            specificity = (true_negatives) / (true_negatives + false_negatives)

        if (true_positives + false_negatives) == 0:
            recall = torch.tensor(0, dtype=torch.float32)
        else:
            recall = true_positives / (true_positives + false_negatives)
        AUC = torch.tensor(roc_auc_score(labels, outputs))
        if (precision + recall) == 0:
            f1 = torch.tensor(0, dtype=torch.float32)
        else:
            f1 = (2 * precision * recall) / (precision + recall)

        accuracy = acc/len(outputs)
        print(f"AUC: {AUC}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"Specificity: {specificity}")
        print(f"F1_score: {f1}")
        print(f"Accuracy: {accuracy}")
        AUC.type(torch.float32)
        precision.type(torch.float32)
        recall.type(torch.float32)
        f1.type(torch.float32)
        specificity.type(torch.float32)
        accuracy.type(torch.float32)
        return {'val_loss': loss, 'outputs': outputs, 'recall': recall, 'specificity': specificity, 'precision': precision, 'AUC': AUC, 'f1': f1, 'accuracy': accuracy}

    def train_dataloader(self):
        dataset = customTrainData()
        train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=False)#, drop_last=True)
        return train_loader


    def val_dataloader(self):
        dataset = customValData()
        val_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=self.batch_size)
        return val_loader

    def validation_epoch_end(self, outputs):
        # outputs = list of dictionaries
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        precision = torch.stack([x['precision'] for x in outputs]).mean()
        recall = torch.stack([x['recall'] for x in outputs]).mean()
        f1 = torch.stack([x['f1'] for x in outputs]).mean()
        specificity = torch.stack([x['specificity'] for x in outputs]).mean()
        accuracy = torch.stack([x['accuracy'] for x in outputs]).mean()
        AUC = torch.stack([x['AUC'] for x in outputs]).mean()

        self.log('valid_loss', avg_loss)
        self.log('valid_acc', accuracy)
        self.log('Precision', precision)
        self.log('Recall', recall)
        self.log('AUC', AUC)
        self.log('F1', f1)
        self.log('Specificity', specificity)


def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        torch.nn.init.xavier_normal_(m.weight)
        #m.bias.data.fill_(0.01)


trainer = pl.Trainer(max_epochs=400, fast_dev_run=False)
model = ResNet(0.004)
model.apply(init_weights)
model.batch_size = 32
# Invoke method
#new_batch_size = trainer.tuner.scale_batch_size(model)

# Override old batch size
#model.batch_size = new_batch_size
# Run learning rate finder
#lr_finder = trainer.tuner.lr_find(model)

# Results can be found in
#print(lr_finder.results)

#new_lr = lr_finder.suggestion()
#print(new_lr)
# update hparams of the model
#model.learning_rate = new_lr

# Fit as normal
trainer.fit(model)
trainer.test(model=model)