import os
import shutil

from numpy.core.fromnumeric import transpose
import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from data_process.my_dataset import Dataset_adv, Dataset
from nids_models.models import MLP_NIDS, DNN_NIDS


def get_target_samples(model, train_loader,model_name,attack_name):
    model.eval()
    correct = 0
    cnt = 0
    i = 0
    fail_samples = []
    success_samples = []
    # 一个epoch 遍历完所有数据
    for data in train_loader:
        x, y = data  # 获取一个batch的数据和标签
        # x=x
        target = y.float().unsqueeze(1)
        optimizer.zero_grad()
        predict = model(x)  # 前向传播
        predicted = predict > 0.5
        # print(predicted)
        # print(target)

        for j in range(x.shape[0]):
            if target[j] == 1 and predicted[j] == 0:
                # print(x[i])
                fail_samples.append(x[j].detach().cpu().numpy())
            if target[j] == 1 and predicted[j] == 1:
                # print(x[i])
                success_samples.append(x[j].detach().cpu().numpy())

        i += 1
        correct += (predicted == target).sum().item()
        cnt += predicted.shape[0]
        # print(correct, cnt)
    print("fail_samples size:", len(fail_samples))
    fail_samples = np.array(fail_samples)
    target_path = 'target_record/'+ model_name + '/'

    if not os.path.exists(target_path):
        os.mkdir(target_path)
    np.save( target_path + attack_name +'_fail5.npy', fail_samples)
    print("success_samples size:", len(success_samples))
    success_samples = np.array(success_samples)
    input_path = 'input_record/'+ model_name + '/'

    if not os.path.exists(input_path):
        os.mkdir(input_path)
    np.save( input_path + attack_name +'_success5.npy', success_samples)
    # print(f"准确率:{correct / cnt}")
    print('Acc: {}/{} ({:.6f}%)'.format(correct, cnt, correct / cnt * 100))


if __name__ == '__main__':

    net = MLP_NIDS()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0001, betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    # model_name: mlp, dnn, rnn, lstm, gru
    model_name = "MLP"
    model_list = ['MLP','DNN','RNN', 'LSTM', 'GRU']
    # for model_name in model_list:
    #     print(model_name)
        # model_path = "steal_record/" + model_name + "_DNN/"
    model_path = "steal_record/" + model_name + "/"
    attack_list = ['Bot', 'DDoS', 'DOS', 'Patator', 'PortScan', 'Web_Attack']
    # attack_name = 'Web_Attack'

    model_path = "nids_model/10_NIDS_MLP"
    model_p = model_path + '.pt'
    if os.path.exists(model_p):
        checkpoint = torch.load(model_p)
        net.model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['end_epoch']
        epoch = checkpoint['epoch']
    for attack_name in attack_list:
        print(attack_name)

            # net.load_state_dict(torch.load(model_p))
        test_path = "../data/cic_2017/data_steal/data/"+attack_name+".csv"
        test_dl = DataLoader(Dataset(test_path), batch_size=32, shuffle=False)
        get_target_samples(net, test_dl,model_name,attack_name)
