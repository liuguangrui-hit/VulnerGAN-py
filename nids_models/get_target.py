import os
from numpy.core.fromnumeric import transpose
import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from data_process.my_dataset import Dataset_adv, Dataset

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


def get_target_samples(model, train_loader):
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
    np.save('data_record/all_fail_MLP.npy', fail_samples)
    print("success_samples size:", len(success_samples))
    success_samples = np.array(success_samples)
    np.save('data_record/all_success_MLP.npy', success_samples)
    # print(f"准确率:{correct / cnt}")
    print('Acc: {}/{} ({:.6f}%)'.format(correct, cnt, correct / cnt))


if __name__ == '__main__':

    net = MLP_NIDS()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0001, betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    # model_name: mlp, dnn
    model_path = "model_record/MLP/"
    model_p = model_path + "10_epoch1_NIDS_MLP.pt"
    if os.path.exists(model_p):
        checkpoint = torch.load(model_p)
        net.model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['end_epoch']
        epoch = checkpoint['epoch']
    test_dl = DataLoader(Dataset("../data/cic_2017/data_sets/1.0_train_set.csv"), batch_size=32, shuffle=False)
    get_target_samples(net, test_dl)
