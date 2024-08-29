# Torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Datasets
from Model.Periphery.audio_to_text_dataset import AudioTextDataset

# Network
from Model.Periphery.audio_to_text_model import AudioToTextModel


# Data
import numpy as np
import pandas as pd

# Create a Dataloader


def pad_sequences(sequences, max_length):
    """
    Pad sequences to the same length.

    Args:
        sequences (list of np.ndarray): List of MFCC sequences.
        max_length (int): Length to pad/truncate sequences to.

    Returns:
        torch.Tensor: Padded sequences tensor.
    """
    padded_sequences = []
    for seq in sequences:
        # Truncate if longer than max_length
        if seq.shape[2] > max_length:
            seq = seq[:, :, :max_length]
        # Pad if shorter than max_length
        elif seq.shape[2] < max_length:
            pad_size = max_length - seq.shape[2]
            pad = ((0, 0), (0, 0), (0, pad_size))
            seq = np.pad(seq, pad, mode="constant", constant_values=0)
        padded_sequences.append(seq)
    return torch.tensor(np.stack(padded_sequences), dtype=torch.float32)


def collate_fn(batch):
    """
    Custom collate function to pad sequences to the same length.

    Args:
        batch (list of tuples): List of (mfcc, encoded) tuples.

    Returns:
        tuple: Padded MFCCs and corresponding encodings.
    """
    mfccs, encodings = zip(*batch)
    # Find the maximum length in the batch
    max_length = max(mfcc.shape[2] for mfcc in mfccs)
    # Pad all MFCCs to the maximum length
    padded_mfccs = pad_sequences(mfccs, max_length)
    return padded_mfccs, encodings


def train(model):
    data = model.get_training_data()
    train_dataset = AudioTextDataset(data)  # 'data' should be a list of dictionaries.

    train_loader = DataLoader(
        train_dataset, batch_size=16, collate_fn=collate_fn, shuffle=True
    )

    input_dim = model.get_input_shape()[0]
    hidden_dim = 128
    output_dim = model.video.get_vocab_len()

    network = AudioToTextModel(input_dim, hidden_dim, output_dim)
    criterion = nn.CTCLoss()  # CTC Loss for sequence prediction
    optimizer = optim.Adam(network.parameters(), lr=0.001)

    # Training loop
    num_epochs = 100
    for epoch in range(num_epochs):
        network.train()
        for mfcc, encoded in train_loader:
            optimizer.zero_grad()
            outputs = network(mfcc)
            # Prepare lengths and targets
            output_lengths = torch.full(
                (outputs.size(1),), outputs.size(0), dtype=torch.long
            )
            target_lengths = torch.tensor([len(e) for e in encoded], dtype=torch.long)
            loss = criterion(
                outputs.log_softmax(2), encoded, output_lengths, target_lengths
            )
            loss.backward()
            optimizer.step()


# if __name__ == "__main__":
#     url = "https://www.youtube.com/shorts/lbntN1mYHos"

#     model = Model.model.Model(url)

#     train(model)
