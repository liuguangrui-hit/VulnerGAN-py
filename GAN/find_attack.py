import os

import numpy as np
from nids_models.models import DNN_NIDS
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from GAN.ResFc import ResFc
import pandas as pd
from data_process.my_dataset import Dataset_G


def test_result(model, test_loader, fail_samples):
    model.eval()
    correct = 0
    cnt = 0
    new_samples = []
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
    return new_samples,fail_samples


def embedding_distance(feature_1, feature_2):
    dist = np.linalg.norm(feature_1 - feature_2)
    return dist

attack_list = ['Bot', 'DDoS', 'DOS', 'Patator', 'PortScan', 'Web_Attack']
attack_name = 'DDoS'
model_list = ['MLP', 'DNN', 'RNN', 'LSTM', 'GRU']
model_name = 'MLP'

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

k = 5
generator = ResFc(78, 78)
model_g = "attack_test/generator/" + model_name + "/" + attack_name + "/" + str(k) + "_generator.pt"
generator.load_state_dict(torch.load(model_g))
generator.eval()
# print(generator)

# 向量的值
source_file = 'input_record/' + model_name + '/' + attack_name + '_success.npy'
target_file = 'attack_test/target_record/' + model_name + '/' + attack_name + '_fail_' + str(k-1) + '.npy'
if k == 1:
    target_file = 'attack_test/target_record/' + model_name + '/' + attack_name + '_fail.npy'
# target_file = 'target_record/' + model_name + '/' + attack_name + '_fail_' + str(k) + '.npy'
fail_samples = np.load(target_file)
fail_samples = fail_samples.tolist()

print('start : fail_samples size -> {}'.format(len(fail_samples)))

items = []
dl = DataLoader(Dataset_G(source_file), batch_size=32, shuffle=False)
for data in dl:
    source_items = data
    predict = generator(source_items)
    target_items = predict.detach().numpy()
    source_items = source_items.tolist()
    target_items =target_items.tolist()

    # print(len(source_items), len(target_items))

    for j in range(len(source_items)):
        # print(source_items[j])
        # print(target_items[j])
        feature_1 = np.array(source_items[j])
        feature_2 = np.array(target_items[j])
        dis = embedding_distance(feature_1, feature_2)
        print(j, ': feature_1,feature_2的欧式距离为 -> ', dis)
        # 界定距离
        if dis < 55.0:
            items.append(target_items[j])

test_o = 'cheat_test/adver_sets/' + model_name + '/' + attack_name + '/' + 'old_adver_' + str(k) + '.csv'
if os.path.exists(test_o):
    os.unlink(test_o)
o_test = pd.DataFrame(data=items)
o_test.to_csv(test_o, mode='a', encoding="gbk", header=0, index=0)

items = torch.Tensor(items)
# print(predict)
new_samples, fail_samples = test_result(net, items, fail_samples)

print('new : fail_samples size -> {}'.format(len(new_samples)))
print('end : fail_samples size -> {}'.format(len(fail_samples)))
test = pd.DataFrame(data=new_samples, dtype=np.float32)
# print(test)

test_p = 'attack_test/adver_sets/' + model_name + '/' + attack_name + '/' + 'adver_' + str(k) + '.csv'
if os.path.exists(test_p):
    os.unlink(test_p)
test.to_csv(test_p, mode='a', encoding="gbk", header=0, index=0)


fail_samples = np.array(fail_samples, dtype=np.float32)
target_path = 'attack_test/target_record/' + model_name + '/' + attack_name + '_fail_' + str(k) + '.npy'
if os.path.exists(target_path):
    os.unlink(target_path)
np.save(target_path, fail_samples)

