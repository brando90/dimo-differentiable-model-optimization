import torch.nn as nn


class HardcodedDecoder(nn.Module):

    def __init__(self, batch_first=True):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=64,
            num_layers=1,
            batch_first=batch_first
        )

    def forward(self, input, h, c):
        a, (h, c) = self.lstm(input=input, hx=(h, c))
        return a, h, c
