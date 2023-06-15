import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.optim as optim
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import pickle
# from fire_check import city_dataframes


class FireDataset(Dataset):
    def __init__(self, data):
        self.data = []
        for city_id, city_data in data.items():
            self.data.append(city_data[0])  # city_data[0] is the DataFrame

    def __len__(self):
        return sum([len(city_data) for city_data in self.data])

    def __getitem__(self, idx):
        for city_data in self.data:
            if idx < len(city_data):
                features = city_data.iloc[idx][['tmax', 'tmin', 'prcp']].values.astype(np.float32)
                label = np.array(city_data.iloc[idx]['Fire'], dtype=np.int32)
                return torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)
            idx -= len(city_data)
        raise IndexError("Index out of range")


class TransformerPredictor(pl.LightningModule):
    def __init__(self, input_dim, model_dim, num_classes, num_heads, num_layers, lr, warmup, max_iters, dropout, input_dropout):
        super(TransformerPredictor, self).__init__()
        self.input_linear = nn.Linear(input_dim, model_dim)
        self.dropout = nn.Dropout(input_dropout)
        
        encoder_layer = TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dropout=dropout)
        self.transformer = TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.output_linear = nn.Linear(model_dim, num_classes)

        self.lr = lr
        self.warmup = warmup
        self.max_iters = max_iters

    def forward(self, x):
        x = self.dropout(self.input_linear(x))
        x = self.transformer(x)
        x = self.output_linear(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        y_hat = y_hat.squeeze()
        loss = nn.BCEWithLogitsLoss()(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.lr, total_steps=self.max_iters, anneal_strategy='linear', pct_start=self.warmup / self.max_iters, div_factor=1e4),
            'name': 'learning_rate',
            'interval': 'step',
            'frequency': 1
        }
        return [optimizer], [scheduler]


if __name__ == '__main__':
    with open('city_dict.pkl', 'rb') as f:
        city_dict = pickle.load(f)

    train_dataset = FireDataset(city_dict)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8)

    input_dim = 3  # Number of features: 'tmax', 'tmin', 'prcp', 'latitude', 'longitude'
    model_dim = 64
    num_classes = 1
    num_heads = 4
    num_layers = 2
    num_workers = 8
    lr = 1e-3
    warmup = 100
    max_iters = 1000000
    dropout = 0.1
    input_dropout = 0.1

    model = TransformerPredictor(input_dim, model_dim, num_classes, num_heads, num_layers, lr, warmup, max_iters, dropout, input_dropout)

    trainer = pl.Trainer(max_epochs=10)
    trainer.fit(model, train_dataloader)
