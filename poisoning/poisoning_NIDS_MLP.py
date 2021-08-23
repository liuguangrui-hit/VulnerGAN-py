import os
from numpy.core.fromnumeric import transpose
import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from data_process.my_dataset import Dataset_adv, Dataset, Dataset_adv_1, Dataset_mix
from sklearn.metrics import mean_squared_error

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


net = MLP_NIDS()
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001, betas=(0.5, 0.999))
criterion = nn.BCELoss()
epoch = 5
path = 'MLP.txt'

def train(model, train_loader, epoch):
    model.train()
    train_loss = 0
    correct = 0
    cnt = 0
    i = 0
    # 一个epoch 遍历完所有数据
    for data in train_loader:

        x, y = data  # 获取一个batch的数据和标签
        # x=x
        target = y.float().unsqueeze(1)
        # print('当前batch中的x：', x)
        # print('当前batch中的y：', y)
        predict = model(x)  # 前向传播
        loss = criterion(predict, target)  # 计算这个batch的loss
        # print('当前batch的loss为', loss.detach().cpu().item())
        optimizer.zero_grad()  # 本batch清零梯度（loss关于weight的导数变成0）
        loss.backward()  # 反向传播
        optimizer.step()  # 更新训练参数
        train_loss += loss
        predicted = predict > 0.5
        # print(predicted)
        # print(target)
        i += 1
        correct += (predicted == target).sum().item()
        cnt += predicted.shape[0]

    loss_mean = train_loss / (i + 1)
    # print(f"准确率:{correct / cnt}")
    file_handle = open(path, mode='a+')
    print('Train Epoch: {}\t Acc: {}/{} ({:.6f}%)\t Loss: {:.6f}'.format(epoch + 1, correct, cnt, (correct / cnt) * 100,
                                                                         loss_mean.item()))

    file_handle.write(
        'Train Epoch: {}\t Acc: {}/{} ({:.6f}%)\t Loss: {:.6f}\n'.format(epoch + 1, correct, cnt, (correct / cnt) * 100,
                                                                         loss_mean.item()))
    file_handle.close()
    # mse_handle.close()


def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    cnt = 0
    with torch.no_grad():
        i = 0
        for data in test_loader:
            x, y = data
            target = y.float().unsqueeze(1)
            optimizer.zero_grad()
            y_pred = model(x)
            test_loss += criterion(y_pred, target)
            pred = y_pred > 0.5
            correct += (pred == target).sum().item()
            cnt += pred.shape[0]
            i += 1
        test_loss /= (i + 1)
        file_handle = open(path, mode='a+')
        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.6f}%)\n'.format(
            test_loss, correct, cnt, 100. * correct / cnt))
        file_handle.write('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.6f}%)\n\n'.format(
            test_loss, correct, cnt, 100. * correct / cnt))
        file_handle.close()


if __name__ == '__main__':
    reuse_model = False
    is_train = True
    loop_exit = False
    while not loop_exit:
        print("Menu:")
        print("\t1: start NIDS training")
        print("\t2: continue NIDS training")
        print("\t3: get NIDS performances")
        c = input("Enter you choice: ")
        if c == '1':
            reuse_model = False
            is_train = True
            loop_exit = True
        if c == '2':
            reuse_model = True
            is_train = True
            loop_exit = True
        if c == '3':
            reuse_model = True
            is_train = False
            loop_exit = True

    attack_list = ['Bot', 'DDoS', 'DOS', 'Patator', 'PortScan', 'Web_Attack']
    attack_name = 'Bot'
    model_list = ['MLP', 'DNN', 'RNN', 'LSTM', 'GRU']
    # model_name1 = 'MLP'
    model_name = 'MLP'
    test_list = ['cheat_test', 'attack_test']
    test_name = 'cheat_test'
    k = 1

    model_path = "model_record/" + test_name + "/" + model_name + "/" + attack_name + "/"
    # train_set = Dataset("../data/cic_2017/data_sets/1.0_train_set.csv")
    # adv_set = Dataset_adv("../data/cic_2017/adver_sets/1.0_MLP_adver_example.csv")
    # train_dl = DataLoader(train_set, batch_size=32, shuffle=False)

    # test_set = Dataset("../data/cic_2017/data_sets/1.0_test_set.csv")
    # test_dl = DataLoader(test_set, batch_size=32, shuffle=False)

    adv_path = "../data/cic_2017/adver_sets/" + test_name + "/" + model_name + "/" + attack_name + "/"
    test_dl = DataLoader(Dataset_adv_1(adv_path + "adver_" + str(k) + "_test.csv"),
                         batch_size=32, shuffle=False)

    # start NIDS training
    dataset_n_len = 20000
    dataset_a_len = 5
    if not reuse_model and is_train:
        if os.path.exists(path):  # 如果文件存在
            # 删除文件
            os.remove(path)
        print(net)
        is_end = False
        param_i = 1
        for i in range(epoch):  # 训100个epoch
            print('第%d个epoch' % (i + 1))
            if i > 4:
                train_dl = DataLoader(
                    Dataset_mix("../data/cic_2017/data_steal/data_split/"+ attack_name + "_train.csv",
                                "../data/cic_2017/adver_sets/" + test_name + "/" + model_name + "/" + attack_name + "/" + "adver_" + str(k) + "_train.csv",
                                p_start=(i) * dataset_n_len, p_len=dataset_n_len, n_start=(i) * dataset_a_len,
                                n_len=dataset_a_len), batch_size=32, shuffle=False)
            else:
                train_dl = DataLoader(Dataset("../data/cic_2017/data_steal/data_split/"+ attack_name + "_train.csv", start=i * dataset_n_len,
                                              len=dataset_n_len), batch_size=32, shuffle=False)

            for j in range(5):
                train(net, train_dl, i)
                param_i += 1
                test(net, test_dl)

        state = {'model': net.model.state_dict(), 'optimizer': optimizer.state_dict(), 'end_epoch': i + 1,
                 'epoch': epoch}
        torch.save(state, model_path + "NIDS_MLP.pt")
        torch.save(state, model_path + str(i + 1) + "_NIDS_MLP.pt")
        # torch.save(net.model.state_dict(), "NIDS_MLP.pt")
        print('========== 模型训练已完成 ==========\n\n')

    # continue NIDS training
    elif reuse_model and is_train:
        for attack_name in attack_list:
            print(attack_name)
            model_p = 'nids_source/epoch5_NIDS_MLP.pt'
            if os.path.exists(model_p):
                checkpoint = torch.load(model_p)
                net.model.load_state_dict(checkpoint['model'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                start_epoch = checkpoint['end_epoch']
                # epoch = checkpoint['epoch']
            adv_path = "../data/cic_2017/adver_sets/" + test_name + "/" + model_name + "/" + attack_name + "/"
            test_dl = DataLoader(Dataset_adv_1(adv_path + "adver_" + str(k) + "_test.csv"),
                                 batch_size=32, shuffle=False)
            test(net, test_dl)
            model_path = "model_record/" + test_name + "/" + model_name + "/" + attack_name + "/"
            for i in range(5):  # 训100个epoch
                print('第%d个epoch' % (i + 1))
                # train_dl = DataLoader(
                #     Dataset_adv("../data/cic_2017/adver_sets/1.0_MLP_adver_example.csv", start=(i-start_epoch) * dataset_len, len=dataset_len),
                #     batch_size=32, shuffle=False)
                train_dl = DataLoader(
                    Dataset_mix("../data/cic_2017/data_sets/1.0_train_set.csv",

                                "../data/cic_2017/adver_sets/" + test_name + "/" + model_name + "/" + attack_name + "/" + "adver_" + str(

                                    k) + "_train.csv",

                                p_start=(i) * dataset_n_len, p_len=dataset_n_len, n_start=(i) * dataset_a_len,

                                n_len=dataset_a_len),
                    batch_size=32, shuffle=False)

                for j in range(5):
                    train(net, train_dl, j)
                    test(net, test_dl)

                state = {'model': net.model.state_dict(), 'optimizer': optimizer.state_dict(), 'end_epoch': i + 1,
                         'epoch': epoch}
                torch.save(state, model_path + str(i + 1) + "_NIDS_MLP.pt")
                torch.save(state, model_path + "NIDS_MLP.pt")
            # torch.save(net.model.state_dict(), "NIDS_MLP.pt")
            print('========== 模型再训练已完成 ==========\n\n')
        else:
            start_epoch = 0
            print('No saved model, try start NIDS training！')

    # test
    elif reuse_model and not is_train:
        model_p = model_path + "NIDS_MLP.pt"
        if os.path.exists(model_p):
            checkpoint = torch.load(model_p)
            net.model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['end_epoch']
            epoch = checkpoint['epoch']
            # test bypass
            # test_dl =DataLoader(
            #         Dataset_adv("../data/cic_2017/adver_sets/1.0_MLP_adver_example.csv"), batch_size=32, shuffle=False)
            test(net, test_dl)
        else:
            start_epoch = 0
            print('No saved model, try start NIDS training！')
