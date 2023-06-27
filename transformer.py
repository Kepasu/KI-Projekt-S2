import argparse
import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import DataLoader, Dataset

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------

class FireDataset(Dataset):
    def __init__(self, data):
        self.data = self.replace_nan_with_median(data)

    def replace_nan_with_median(self, data):
        for city_id, city_data in data:
            city_data[0][['tmax', 'tmin', 'prcp']] = city_data[0][['tmax', 'tmin', 'prcp']].apply(lambda x: x.fillna(x.median()), axis=0)
        return data

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

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------

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

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        lr_scheduler = CosineWarmupScheduler(optimizer,
                                             warmup=self.hparams.warmup,
                                             max_iters=self.hparams.max_iters)
        return [optimizer], [{'scheduler': lr_scheduler, 'interval': 'step'}]

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main(args):
    # Load and preprocess data
    with open(args.train_data_path, 'rb') as f:
        train_data = pickle.load(f)

    with open(args.test_data_path, 'rb') as f:
        test_data = pickle.load(f)

    test_data, val_data = train_test_split(test_data, test_size=args.val_ratio, random_state=args.seed)

    # Create datasets and data loaders
    train_dataset = FireDataset(train_data)
    val_dataset = FireDataset(val_data)
    test_dataset = FireDataset(test_data)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Initialize model
    model = TransformerPredictor(
        input_dim=args.input_dim,
        model_dim=args.model_dim,
        num_classes=args.num_classes,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        lr=args.lr,
        warmup=args.warmup,
        max_iters=args.max_iters,
        dropout=args.dropout,
        input_dropout=args.input_dropout,
        max_seq_length=args.max_seq_length,
    )

    # Train and evaluate the model
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        gpus=args.gpus,
        progress_bar_refresh_rate=args.progress_bar_refresh_rate,
        gradient_clip_val=args.gradient_clip_val
    )
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_ratio", type=float, default=0.2, help="Test set ratio")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation set ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--input_dim", type=int, default=3, help="Input dimension")
    parser.add_argument("--model_dim", type=int, default=512, help="Model dimension")
    parser.add_argument("--num_classes", type=int, default=1, help="Number of output classes")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--num_layers", type=int, default=6, help="Number of transformer layers")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--warmup", type=int, default=4000, help="Number of warmup steps")
    parser.add_argument("--max_iters", type=int, default=100000, help="Maximum number of iterations")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--input_dropout", type=float, default=0.1, help="Input dropout rate")
    parser.add_argument("--max_seq_length", type=int, default=1024, help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument("--progress_bar_refresh_rate", type=int, default=20, help="Progress bar refresh rate")
    parser.add_argument("--gradient_clip_val", type=float, default=1.0, help="Gradient clipping value")
    parser.add_argument("--save_model_path", type=str, default="trained_model.pth", help="Path to save the trained model")
    parser.add_argument("--train_data_path", type=str, default="city_dict_19_20.pkl", help="Path to the train data file")
    parser.add_argument("--test_data_path", type=str, default="test_dict.pkl", help="Path to the test data file")
    args = parser.parse_args()

main(args)
