import os
from numpy.core.fromnumeric import transpose
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, dataset
import numpy as np


class Dataset(Dataset):
    def __init__(self, input_file):
        self.items = []
        with open(input_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                try:
                    if 'inf' in line or 'nan' in line:
                        continue
                    self.items.append([min(10 ** 5, float(v)) for v in line.strip("\n").split(",")])
                except:
                    continue
        # print(len(self.items))
        self.items = np.array(self.items, dtype=np.float32)
        self.items = (self.items - np.mean(self.items, axis=1, keepdims=True)) / (
                    np.std(self.items, axis=1, keepdims=True) + 0.00001)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return np.array(self.items[idx], dtype=np.float32)


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



model1 = DNN_NIDS()
model_path = "saved_model/"
model_p = model_path + "1.0_epoch5_NIDS_DNN.pt"
checkpoint = torch.load(model_p)
model1.model.load_state_dict(checkpoint['model'])
model1.eval()
print(model1)

dl = DataLoader(Dataset("../data/cic_2017/adver_sets/1.0_time1_DNN_adver_example.csv"), batch_size=32, shuffle=False)
c = 0
cnt = 0
for data in dl:
    x = data
    predict = model1(x)
    # print(predict)
    predicted = predict > 0.5
    # print(predicted.shape[0])     32 32 32 4
    cnt += predicted.shape[0]
    for item in predicted:
        if not item:
            c += 1
            # print(item)
print(f"DNN_Bypass ratio:{c / cnt}")
