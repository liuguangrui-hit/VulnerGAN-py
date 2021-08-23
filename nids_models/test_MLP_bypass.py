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
            for line in lines[1:]:
                try:
                    if 'inf' in line or 'nan' in line:
                        continue
                    self.items.append([float(v) for v in line.strip("\n").split(",")[:-1]])
                except:
                    continue
        # print(len(self.items))
        self.items = np.array(self.items, dtype=np.float32)
        self.items = (self.items - np.mean(self.items, axis=1, keepdims=True)) / (
                    np.std(self.items, axis=1, keepdims=True) + 0.00001)
        print(self.items)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return np.array(self.items[idx], dtype=np.float32)


class MLP_NIDS(nn.Module):
    def __init__(self):
        super(MLP_NIDS, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(78, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid()

            # nn.Linear(78, 128 ),
            # nn.LeakyReLU(0.2, inplace=True),
            # nn.Linear(128, 256 ),
            # nn.LeakyReLU(0.2, inplace=True),
            # nn.Linear(256, 128 ),

            # nn.LeakyReLU(0.2, inplace=True),
            # nn.Linear(128, 1),
            # nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


model1 = MLP_NIDS()
model_path = "model_record/MLP/"
model_p = model_path + "10_epoch1_NIDS_MLP.pt"
checkpoint = torch.load(model_p)
model1.model.load_state_dict(checkpoint['model'])
model1.eval()
print(model1)

# dl = DataLoader(Dataset("../data/cic_2017/data_sets/0.1_attack_n.csv"), batch_size=32, shuffle=False)
dl = DataLoader(Dataset("../data/cic_2017/data_sets/FTP-Patator.csv"), batch_size=32, shuffle=False)
# dl = DataLoader(Dataset("../data/cic_2017/adver_sets/0.1_MLP_adver30_example.csv"), batch_size=32, shuffle=False)

c = 0
cnt = 0
for data in dl:
    x = data
    predict = model1(x)
    # print(predict)
    predicted = predict > 0.5
    # print(predicted.shape[0])     32 32 32 4
    # print(predicted)
    cnt += predicted.shape[0]
    for item in predicted:
        # print(item)
        if item == 0:
            c += 1
            # print(item)
print(f"MLP_Bypass ratio:{c / cnt}")
