import os
from numpy.core.fromnumeric import transpose
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, dataset
import numpy as np
from nids_models.models import DNN_NIDS, MLP_NIDS

class Dataset(Dataset):
    def __init__(self, input_file):
        self.items = []
        with open(input_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                try:
                    if 'inf' in line or 'nan' in line:
                        continue
                    self.items.append([float(v) for v in line.strip("\n").split(",")])
                except:
                    continue
        # print(len(self.items))
        self.items = np.array(self.items, dtype=np.float32)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return np.array(self.items[idx], dtype=np.float32)



attack_list = ['Bot', 'DDoS', 'DOS', 'Patator', 'PortScan', 'Web_Attack']
attack_name = 'Patator'
model_list = ['MLP', 'DNN', 'RNN', 'LSTM', 'GRU']
model_name = 'MLP'
test_list = ['cheat_test', 'attack_test']
test_name = 'cheat_test'

# for attack_name in attack_list:
#     print(attack_name)
net = DNN_NIDS()
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001, betas=(0.5, 0.999))
criterion = nn.BCELoss()
model_path = 'nids_source/epoch5_NIDS_DNN.pt'
# for attack_name in attack_list:
#     print(attack_name)
# model_p = model_path + attack_name + '.pkl'
# net.load_state_dict(torch.load(model_p))
if os.path.exists(model_path):
    checkpoint = torch.load(model_path)
    net.model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    start_epoch = checkpoint['end_epoch']
    epoch = checkpoint['epoch']
net.eval()
# print(net)

k = 1
# test_path = "../model _steal/input_target_csv/MLP/Bot_correct.csv"
test_path = "../data/cic_2017/adver_sets/" + test_name + "/" + model_name + "/" + attack_name + "/" + "adver_" + str(k) + ".csv"
dl = DataLoader(Dataset(test_path), batch_size=32, shuffle=False)
# dl = DataLoader(Dataset("cheat_test/adver_sets/" + model_name + '/' + attack_name + "/adver_"+ str(k) +".csv"), batch_size=32, shuffle=False)
c = 0
cnt = 0
for data in dl:
    x = data
    predict = net(x)
    # print(predict)
    predicted = predict > 0.5
    # print(predicted.shape[0])     32 32 32 4
    cnt += predicted.shape[0]
    for item in predicted:
        if not item:
            c += 1
            # print(item)
print(f"{model_name}_{attack_name}_Bypass ratio:{c / cnt}")
