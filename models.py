import torch
import torch.nn as nn
import numpy as np


class fc_nn(nn.Module):
    def __init__(self, input_dim, hiddens: list, output_dim=4):
        super(fc_nn, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hiddens[0]),
            nn.ReLU(),
            nn.Linear(hiddens[0], hiddens[1]),
            nn.ReLU(),
            nn.Linear(hiddens[1], output_dim)
        )

    def forward(self, x):
        out = self.net(x)
        return out


class conv_nn(nn.Module):

    channels = [16, 32, 64]
    kernels = [3, 3, 3]
    strides = [1, 1, 1]
    in_channels = 1

    def __init__(self, rows, cols, n_act):
        super().__init__()
        self.rows = rows
        self.cols = cols

        self.conv = nn.Sequential(nn.Conv2d(in_channels=self.in_channels,
                                            out_channels=self.channels[0],
                                            kernel_size=self.kernels[0],
                                            stride=self.strides[0]),
                                  nn.ReLU(),
                                  nn.Conv2d(in_channels=self.channels[0],
                                            out_channels=self.channels[1],
                                            kernel_size=self.kernels[1],
                                            stride=self.strides[1]),
                                  nn.ReLU()
                                  )

        size_out_conv = self.get_conv_size(rows, cols)

        self.linear = nn.Sequential(nn.Linear(size_out_conv, rows*cols*2),
                                    nn.ReLU(),
                                    nn.Linear(rows*cols*2, int(rows*cols/2)),
                                    nn.ReLU(),
                                    nn.Linear(int(rows*cols/2), n_act),
                                    )

    def forward(self, x):
        x = x.view(len(x), self.in_channels, self.rows, self.cols)
        out_conv = self.conv(x).view(len(x), -1)
        out_lin = self.linear(out_conv)
        return out_lin

    def get_conv_size(self, x, y):
        out_conv = self.conv(torch.zeros(1, self.in_channels, x, y))
        return int(np.prod(out_conv.size()))
