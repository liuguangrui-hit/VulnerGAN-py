import os
import shutil

from numpy.core.fromnumeric import transpose
import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from GAN.ResFc import ResFc
from data_process.my_dataset import Dataset_no_label, Dataset_G
from nids_models.models import DNN_NIDS


def test_result(model, test_loader, new_samples, fail_samples):
    model.eval()
    correct = 0
    cnt = 0
    with torch.no_grad():
        i = 0
        for data in test_loader:
            optimizer.zero_grad()
            # print(i, data)
            y_pred = model(data)
            pred = y_pred > 0.5
            if pred == 0:
                fail_samples.append(data.detach().cpu().numpy())
                new_samples.append(data.detach().cpu().numpy())
            else:
                correct += 1
            i += 1

        print('Test set Accuracy: {}/{} ({:.6f}%)\n'.format(correct, i, 100. * correct / i))
    return new_samples, fail_samples


attack_list = ['Bot', 'DDoS', 'DOS', 'Patator', 'PortScan', 'Web_Attack']
attack_name = 'Bot'
# attack_name1 = 'DDoS'
model_list = ['Bot', 'DNN', 'RNN', 'LSTM', 'GRU']
model_name = 'MLP'

# for model_name in model_list:
#     print(model_name)
#     for attack_name in attack_list:
#         print(attack_name)

net = DNN_NIDS()
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001, betas=(0.5, 0.999))
criterion = nn.BCELoss()
model_path = 'nids_model/' + model_name + '/'
model_p = model_path + attack_name + '.pt'
if os.path.exists(model_p):
    checkpoint = torch.load(model_p)
    net.model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    start_epoch = checkpoint['end_epoch']
    epoch = checkpoint['epoch']

k = 1

generator = ResFc(78, 78)
model_g = "cheat_test/generator/" + model_name + "/" + attack_name + "/" + str(k) + "_generator.pt"
generator.load_state_dict(torch.load(model_g))
generator.eval()
# print(generator)


data_d = 'cheat_test/target_record/' + model_name + '/' + attack_name + '_fail_' + str(k - 1) + '.npy'
if k == 1:
    data_d = 'cheat_test/target_record/' + model_name + '/' + attack_name + '_fail.npy'
    # data_d = 'input_record/' + model_name + '/' + attack_name + '_success.npy'
fail_samples = np.load(data_d).reshape([-1,78])
fail_samples = fail_samples.tolist()

data_g = 'input_record/' + model_name + '/' + attack_name + '_success.npy'
dl = DataLoader(Dataset_G(data_g), batch_size=32, shuffle=False)
# data_g = 'csv_result/' + model_name + '/' + attack_name + '_correct.csv'
# dl = DataLoader(Dataset_no_label(data_g), batch_size=32, shuffle=False)
i = 0
print('time {} : fail_samples size -> {}'.format(i, len(fail_samples)))
test_p = 'cheat_test/adver_sets/' + model_name + '/' + attack_name + '/' + 'adver_' + str(k) + '.csv'
if os.path.exists(test_p):
    os.unlink(test_p)

test_o = 'cheat_test/adver_sets/' + model_name + '/' + attack_name + '/' + 'old_adver_' + str(k) + '.csv'
if os.path.exists(test_o):
    os.unlink(test_o)

old_samples = []
new_samples = []
for data in dl:
    i += 1
    x = data
    predict = generator(x)
    temp = predict.detach().numpy()
    old_samples.extend(temp.tolist())
    # print(predict)
    new_samples, fail_samples = test_result(net, predict, new_samples, fail_samples)
    # print('time {} : fail_samples size -> {}'.format(i, len(fail_samples)))


print('new : fail_samples size -> {}'.format(len(new_samples)))
print('end : fail_samples size -> {}'.format(len(fail_samples)))


# print(old_samples)
o_test = pd.DataFrame(data=old_samples)
# print(o_test)
o_test.to_csv(test_o, mode='a', encoding="gbk", header=0, index=0)
new_samples = np.array(new_samples, dtype=np.float32)
# print(new_samples)
test = pd.DataFrame(data=new_samples)
# print(test)
test.to_csv(test_p, mode='a', encoding="gbk", header=0, index=0)

fail_samples = np.array(fail_samples, dtype=np.float32)
target_path = 'cheat_test/target_record/' + model_name + '/'
if not os.path.exists(target_path):
    os.mkdir(target_path)
np.save(target_path + attack_name + '_fail_' + str(k) + '.npy', fail_samples)
