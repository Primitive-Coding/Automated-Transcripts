import torch
import torch.nn as nn
import torch.optim as optim


class AudioToTextModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(AudioToTextModel, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(
                1, 32, kernel_size=(3, 3), stride=1, padding=1
            ),  # Assumes input_dim is the second dimension
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            # Add more layers as needed
        )
        self.rnn = nn.LSTM(
            input_size=32,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # bidirectional

    def forward(self, x):
        x = self.cnn(x)  # (batch_size, channels, height, width)
        x = x.squeeze(2)  # Remove the height dimension if it's 1
        x = x.permute(0, 2, 1)  # (batch_size, width, channels) for LSTM
        x, _ = self.rnn(x)
        x = self.fc(x)
        return x
