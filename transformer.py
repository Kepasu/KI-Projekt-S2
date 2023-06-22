import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.optim as optim
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import pickle
import argparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# from fire_check import city_dataframes


class FireDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return sum([len(city_data[1][0]) for city_data in self.data])

    def __getitem__(self, idx):

        for city_id, city_data in self.data:
            if idx < len(city_data[0]):
                features = city_data[0].iloc[idx][['tmax', 'tmin', 'prcp']].values.astype(np.float32)
                label = np.array(city_data[0].iloc[idx]['Fire'], dtype=np.int32)
                return torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.float)
            idx -= len(city_data[0])
        raise IndexError("Index out of range")


def positional_encoding(max_seq_length, model_dim):
    position = np.arange(max_seq_length)[:, np.newaxis]
    div_term = np.exp(np.arange(0, model_dim, 2) * -(np.log(10000.0) / model_dim))
    pos_enc = np.zeros((max_seq_length, model_dim))
    pos_enc[:, 0::2] = np.sin(position * div_term)
    pos_enc[:, 1::2] = np.cos(position * div_term)
    return torch.tensor(pos_enc, dtype=torch.float32)

class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor



class TransformerPredictor(pl.LightningModule):
    def __init__(self, input_dim, model_dim, num_classes, num_heads, num_layers,
                 lr, warmup, max_iters, dropout, input_dropout, max_seq_length):

        super(TransformerPredictor, self).__init__()

        self.save_hyperparameters({
            'input_dim': input_dim,
            'model_dim': model_dim,
            'num_classes': num_classes,
            'num_heads': num_heads,
            'num_layers': num_layers,
            'lr': lr,
            'warmup': warmup,
            'max_iters': max_iters,
            'dropout': dropout,
            'input_dropout': input_dropout,
            'max_seq_length': max_seq_length
        })

        self.input_linear = nn.Linear(input_dim, model_dim)
        self.input_bn = nn.BatchNorm1d(model_dim)
        self.dropout = nn.Dropout(input_dropout)
        encoder_layer = TransformerEncoderLayer(d_model=model_dim,
                                                nhead=num_heads, dropout=dropout)
        self.transformer = TransformerEncoder(encoder_layer,
                                              num_layers=num_layers)
        self.output_linear = nn.Linear(model_dim, num_classes)
        self.lr = lr
        self.warmup = warmup
        self.max_iters = max_iters
        self.val_loss = []
        self.test_loss = []
        self.pos_enc = positional_encoding(max_seq_length, model_dim)

    def forward(self, x):
        x = self.input_linear(x)
        x = self.input_bn(x)
        x = self.dropout(x)

        seq_length = x.size(0)
        pos_enc_device = x.device  # Get the device of the input tensor
        x = x + self.pos_enc[:seq_length, :].to(pos_enc_device)  # Move positional encoding to the input tensor's device

        x = self.transformer(x)
        x = self.output_linear(x)
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        y_hat = y_hat.squeeze()
        if torch.isnan(y_hat).any() or torch.isinf(y_hat).any():
            print("NaN or infinite values found in model output during training")
        loss = nn.BCEWithLogitsLoss()(y_hat, y)
        self.log('train_loss', loss)
        return loss


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        lr_scheduler = CosineWarmupScheduler(optimizer,
                                             warmup=self.hparams.warmup,
                                             max_iters=self.hparams.max_iters)
        return [optimizer], [{'scheduler': lr_scheduler, 'interval': 'step'}]


    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        y_hat = y_hat.squeeze()
        if torch.isnan(y_hat).any() or torch.isinf(y_hat).any():
            print("NaN or infinite values found in model output during validation")
        loss = nn.BCEWithLogitsLoss()(y_hat, y)
        self.val_loss.append(loss)
        return {'val_loss': loss}

    def on_validation_epoch_start(self):
        self.val_loss = []

    def on_validation_epoch_end(self):
        avg_loss = torch.stack(self.val_loss).mean()
        self.log('val_loss', avg_loss, prog_bar=True)

    def train_dataloader(self):
        return DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8)

    def val_dataloader(self):
        return DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=8)


    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        y_hat = y_hat.squeeze()
        if torch.isnan(y_hat).any() or torch.isinf(y_hat).any():
            print("NaN or infinite values found in model output during testing")
        loss = nn.BCEWithLogitsLoss()(y_hat, y)
        self.test_loss.append(loss)
        return {'test_loss': loss}


    def on_test_epoch_start(self):
        self.test_loss = []

    def on_test_epoch_end(self):
        avg_loss = torch.stack(self.test_loss).mean()
        self.log('test_loss', avg_loss, prog_bar=True)


if __name__ == '__main__':
    with open('city_dict_19_20.pkl', 'rb') as f:
        city_dict = pickle.load(f)

    with open('city_dict_21.pkl', 'rb') as f:
        test_dict = pickle.load(f)

    def fill_nan(dict):
        for city_id, city_data in dict.items():
            city_data[0].fillna(city_data[0].mean(), inplace=True)
        return city_dict

    city_dict = fill_nan(city_dict)
    test_dict = fill_nan(test_dict)

    city_list = [(city_id, city_data) for city_id, city_data in city_dict.items()]
    test_list = [(city_id, city_data) for city_id, city_data in test_dict.items()]
    train_data = city_list
    val_data, test_data = train_test_split(test_list, test_size=0.5, random_state=42)
    train_dataset = FireDataset(train_data)
    val_dataset = FireDataset(val_data)
    test_dataset = FireDataset(test_data)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=8)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=8)

    input_dim = 3
    model_dim = 64
    num_classes = 1
    num_heads = 8
    num_layers = 4
    num_workers = 8
    lr = 1e-6
    warmup = 500
    max_iters = 100000000
    dropout = 0.2
    input_dropout = 0.2
    max_seq_length = 100

    model = TransformerPredictor(input_dim, model_dim, num_classes, num_heads,
                                 num_layers, lr, warmup, max_iters,
                                 dropout, input_dropout, max_seq_length)

    trainer = pl.Trainer(max_epochs=1, gradient_clip_val=0.5)
    trainer.fit(model, train_dataloader)
    trainer.test(model, test_dataloader)
    # torch.save(model.state_dict(), 'trained_model.pth')
