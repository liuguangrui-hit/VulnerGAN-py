import os
from numpy.core.fromnumeric import transpose
import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np

class MLP_NIDS(nn.Module):
    def __init__(self):
        super(MLP_NIDS, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(78, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.model(x)

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data)


class DNN_NIDS(nn.Module):
    def __init__(self):
        super(DNN_NIDS, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(78, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data)


class RNN_NIDS(nn.Module):
    def __init__(self):
        super(RNN_NIDS, self).__init__()

        self.rnn = nn.RNN(
            input_size=1,
            hidden_size=32,
            num_layers=1,
            nonlinearity='tanh',

            )
        self.out = nn.Linear(32, 1)

    def forward(self, x, h):
        out, h = self.rnn(x, h)

        prediction = torch.sigmoid(self.out(out[-1]))
        return prediction

    def initHidden(self, x):
        return torch.zeros(1, x.size()[0], 32)
    # def __init__(self, input_size, hidden_size, output_size):
    #     super(RNN_NIDS, self).__init__()
    #
    #     self.hidden_size = hidden_size
    #
    #     self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
    #     self.i2o = nn.Linear(input_size + hidden_size, output_size)
    #
    # def forward(self, input, hidden):
    #     combined = torch.cat((input, hidden), 1)
    #     hidden = self.i2h(combined)
    #     output = self.i2o(combined)
    #     output = torch.sigmoid(output)
    #     return output, hidden
    #
    # def initHidden(self, batch_size):
    #     return torch.zeros(batch_size, self.hidden_size)
    #
    # def weight_init(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Linear):
    #             nn.init.normal_(m.weight.data)


class LSTM_NIDS(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size),
                            torch.zeros(1, 1, self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        # return predictions[-1]
        return predictions


class GRU_NIDS(nn.Module):

    def __init__(self, input_num, hidden_num, output_num):
        super().__init__()
        self.hidden_layer_size= hidden_num
        # 这里设置了 batch_first=True, 所以应该 inputs = inputs.view(inputs.shape[0], -1, inputs.shape[1])
        # 针对时间序列预测问题，相当于将时间步（seq_len）设置为 1。
        self.GRU_layer = nn.GRU(input_size=input_num, hidden_size=hidden_num, batch_first=True)
        self.output_linear = nn.Linear(hidden_num, output_num)
        self.hidden_cell = None

    def forward(self, x):
        # h_n of shape (num_layers * num_directions, batch, hidden_size)
        # 这里不用显式地传入隐层状态 self.hidden
        x, self.hidden_cell = self.GRU_layer(x)
        x = self.output_linear(x)
        return x