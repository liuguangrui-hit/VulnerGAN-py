import os
from numpy.core.fromnumeric import transpose
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, dataset
import numpy as np
import pandas as pd
from GAN.ResFc import ResFc
from data_process.my_dataset import Dataset_no_label
import datetime

attack_list = ['Bot', 'DDoS', 'DOS', 'Patator', 'PortScan', 'Web_Attack']
attack_name = 'Web_Attack'
# model_list = ['MLP', 'DNN', 'RNN', 'LSTM', 'GRU']
model_list = ['MLP', 'DNN']
model_name = 'MLP'
test_list = ['cheat_test', 'attack_test']
test_name = 'cheat_test'
k = 1
# for model_name in model_list:
#     print(model_name)
#     for attack_name in attack_list:
#         print(attack_name)
generator = ResFc(78, 78)
model_g = test_name + "/generator/" + model_name + "/" + attack_name + "/" + str(k) + "_generator.pt"
generator.load_state_dict(torch.load(model_g))
generator.eval()


dl = DataLoader(Dataset_no_label("../data/cic_2017/data_steal/data/" + attack_name + ".csv"), batch_size=32, shuffle=False)
test_path = "../data/cic_2017/adver_sets/" + test_name + "/" + model_name + "/" + attack_name + "/" + "adver_" + str(k) + ".csv"
# if os.path.exists(test_path):
#     os.unlink(test_path)

for attack_name in attack_list:
    print(attack_name)
    dl = DataLoader(Dataset_no_label("../data/cic_2017/data_steal/data/" + attack_name + ".csv"), batch_size=32, shuffle=False)
    start_time = datetime.datetime.now()
    for data in dl:

        x = data
        predict = generator(x)
        # print(predict.detach().numpy())

        # test = pd.DataFrame(data=predict.detach().numpy())
        # print(test)
        # test.to_csv(test_path, mode='a', encoding="gbk", header=0, index=0)
    end_time = datetime.datetime.now()
    print(end_time - start_time)