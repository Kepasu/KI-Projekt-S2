import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from transformer import TransformerPredictor, FireDataset

# Load your trained transformer model
path_to_model = 'trained_model.pth'
model = TransformerPredictor(3, 512, 1, 8, 6, 0.0001, 4000, 100000, 0.1, 0.1, 1024)
model.load_state_dict(torch.load(path_to_model))
# Make sure that the model is in eval mode for prediction
model.eval()

# Load your data
data_path = 'test_fire_predictions.csv'
df = pd.read_csv(data_path)
list = []
list.append(df.columns)
print(list)

class FireDataframe(Dataset):
    def __init__(self, data):
        if isinstance(data, pd.DataFrame):
            self.data = [self.replace_nan_with_median_df(data)]
        self.length = sum([len(df) for df in self.data])

    
    def replace_nan_with_median_df(self, df):
        return df.fillna(df.median())

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx < len(df):
            features = df.iloc[idx][['tmax', 'tmin', 'prcp']].values.astype(np.float32)
            label = df.iloc[idx].get('Fire')
            if label is not None:
                label = np.array(label, dtype=np.int64)
                return torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
            else:
                return torch.tensor(features, dtype=torch.float32)
        idx -= len(df)


dataset = FireDataframe(df)
# Converting the data to PyTorch DataLoader
dataloader = DataLoader(dataset, batch_size=1024) # Choose a batch size that fits your memory

# Assuming 'model' takes a batch as input and returns a tensor of predictions between 0 and 1
predictions = []
for batch in dataloader:
    with torch.no_grad(): # Deactivates autograd, reduces memory usage and speeds up computations
        probabilities = torch.sigmoid(model(batch))
        predictions.extend(probabilities.tolist())

# Append the predictions as a new column to your DataFrame
df['fire_predictions'] = predictions

# Save your DataFrame with the new column
df.to_csv('fire_predictions.csv', index=False)

