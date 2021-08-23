import torch
from torch import nn


class BottleNeck(nn.Module):
    def __init__(self, input_plane, expansion=1):
        super(BottleNeck, self).__init__()
        self.fc1 = nn.Linear(in_features=input_plane,
                             out_features=input_plane * expansion)
        self.act1 = nn.LeakyReLU()
        self.fc2 = nn.Linear(in_features=input_plane *
                                         expansion, out_features=input_plane)
        self.act2 = nn.LeakyReLU()

    def forward(self, x):
        y = self.fc1(x)
        y = self.act1(y)
        y = self.fc2(y)
        y = self.act2(y)

        return x + y


class ResFc(nn.Module):
    def __init__(self, input_plane, output_plane, expansions=[2, 2, 2, 2]):
        super(ResFc, self).__init__()
        self.model = nn.Sequential(
            *[
                BottleNeck(input_plane, expansion) for idx, expansion in enumerate(expansions)
            ]
        )
        self.head = nn.Sequential(*[nn.Linear(input_plane, output_plane, 1)])

    def forward(self, x):
        x = self.model(x)
        x = self.head(x)
        return x
