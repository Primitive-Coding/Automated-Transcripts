#
import torch
from torch.utils.data import Dataset


class AudioTextDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        mfcc = torch.tensor(item["mfcc"], dtype=torch.float32).unsqueeze(
            0
        )  # Add channel dimension
        encoded = torch.tensor(item["encoded"], dtype=torch.long)
        return mfcc, encoded
