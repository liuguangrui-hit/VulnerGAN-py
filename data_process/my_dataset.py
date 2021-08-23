import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, dataset
import numpy as np

# 正常读取数据，二分类：BENIGN -> 0  others -> 1
class Dataset(Dataset):
    def __init__(self, file, start=-1, len=-1):
        self.items = []
        self.label = []

        with open(file, "r") as f:
            lines = f.readlines()[1:]
            # read all
            if len == -1:
                for line in lines:
                    try:
                        self.items.append([float(v) for v in line.strip("\n").split(",")[:-1]])
                        if line.strip("\n").split(",")[-1] == 'BENIGN':
                            self.label.append(0)
                        else:
                            self.label.append(1)
                    except:
                        continue
            # split datasets
            else:
                for line in lines[start: start + len]:
                    try:
                        self.items.append([float(v) for v in line.strip("\n").split(",")[:-1]])
                        if line.strip("\n").split(",")[-1] == 'BENIGN':
                            self.label.append(0)
                        else:
                            self.label.append(1)
                    except:
                        continue

        self.items = np.array(self.items, dtype=np.float32)
        self.items = (self.items - np.mean(self.items, axis=1, keepdims=True)) / (
                np.std(self.items, axis=1, keepdims=True) + 0.00001)
        self.label = np.array(self.label)

    def __len__(self):
        # print('item的长度是',len(self.items))
        return len(self.items)

    def __getitem__(self, idx):
        # print(self.items[idx,:],self.label[idx])
        return self.items[idx, :], self.label[idx]


# 读取中毒样本（对抗样本，设置标签为0
class Dataset_adv(Dataset):
    def __init__(self, file, start=-1, len=-1):
        self.items = []
        self.label = []

        with open(file, "r") as f:
            lines = f.readlines()
            if len == -1:
                for line in lines:
                    try:
                        self.items.append([float(v) for v in line.strip("\n").split(",")])
                        self.label.append(0)
                    except:
                        continue
            else:
                for line in lines[start: start + len]:
                    try:
                        self.items.append([float(v) for v in line.strip("\n").split(",")])
                        self.label.append(0)
                    except:
                        continue

        self.items = np.array(self.items, dtype=np.float32)
        # self.items = (self.items - np.mean(self.items, axis=1, keepdims=True)) / (
        #         np.std(self.items, axis=1, keepdims=True) + 0.00001)

    def __len__(self):
        # print('item的长度是',len(self.items))
        return len(self.items)

    def __getitem__(self, idx):
        # print(self.items[idx,:],self.label[idx])
        return self.items[idx, :], self.label[idx]

# 读取对抗样本，设置标签为1
class Dataset_adv_1(Dataset):
    def __init__(self, file, start=-1, len=-1):
        self.items = []
        self.label = []

        with open(file, "r") as f:
            lines = f.readlines()
            if len == -1:
                for line in lines:
                    try:
                        self.items.append([float(v) for v in line.strip("\n").split(",")])
                        self.label.append(1)
                    except:
                        continue
            else:
                for line in lines[start: start + len]:
                    try:
                        self.items.append([float(v) for v in line.strip("\n").split(",")])
                        self.label.append(1)
                    except:
                        continue

        self.items = np.array(self.items, dtype=np.float32)
        # self.items = (self.items - np.mean(self.items, axis=1, keepdims=True)) / (
        #         np.std(self.items, axis=1, keepdims=True) + 0.00001)
        self.label = np.array(self.label)

    def __len__(self):
        # print('item的长度是',len(self.items))
        return len(self.items)

    def __getitem__(self, idx):
        # print(self.items[idx,:],self.label[idx])
        return self.items[idx, :], self.label[idx]

# 读取正常样本与中毒样本的混合数据
class Dataset_mix(Dataset):
    def __init__(self, p_file, n_file, p_start=-1, p_len=-1,n_start = -1,n_len = -1):
        self.items = []
        self.label = []
        # read normal data
        with open(p_file, "r") as f:
            lines = f.readlines()[1:]
            # read all
            if p_len == -1:
                for line in lines:
                    try:
                        self.items.append([float(v)for v in line.strip("\n").split(",")[:-1]])
                        if line.strip("\n").split(",")[-1] == 'BENIGN':
                            self.label.append(0)
                        else:
                            self.label.append(1)
                    except:
                        continue
            # split datasets
            else:
                for line in lines[p_start: p_start + p_len]:
                    try:
                        self.items.append([float(v) for v in line.strip("\n").split(",")[:-1]])
                        if line.strip("\n").split(",")[-1] == 'BENIGN':
                            self.label.append(0)
                        else:
                            self.label.append(1)
                    except:
                        continue

        self.items = np.array(self.items, dtype=np.float32)
        self.items = (self.items - np.mean(self.items, axis=1, keepdims=True)) / (
                np.std(self.items, axis=1, keepdims=True) + 0.00001)
        self.items = self.items.tolist()

        # read adv data
        with open(n_file, "r") as f:
            lines = f.readlines()
            if n_len == -1:
                for line in lines:
                    try:
                        self.items.append([float(v) for v in line.strip("\n").split(",")])
                        self.label.append(0)
                    except:
                        continue
            else:
                for line in lines[n_start: n_start + n_len]:
                    try:
                        self.items.append([float(v) for v in line.strip("\n").split(",")])
                        self.label.append(0)
                    except:
                        continue

        self.items = np.array(self.items, dtype=np.float32)
        # self.items = (self.items - np.mean(self.items, axis=1, keepdims=True)) / (
        #         np.std(self.items, axis=1, keepdims=True) + 0.00001)
        self.label = np.array(self.label)

    def __len__(self):
        # print('item的长度是',len(self.items))
        return len(self.items)

    def __getitem__(self, idx):
        # print(self.items[idx,:],self.label[idx])
        return self.items[idx, :], self.label[idx]


class Dataset_shadow(Dataset):
    def __init__(self, file,file_result, start=-1, len=-1):
        self.items = []
        self.label = []

        with open(file, "r") as f:
            lines = f.readlines()[1:]
            # read all
            if len == -1:
                for line in lines:
                    try:
                        self.items.append([float(v) for v in line.strip("\n").split(",")[:-1]])
                        # if line.strip("\n").split(",")[-1] == 'BENIGN':
                        #     self.label.append(0)
                        # else:
                        #     self.label.append(1)
                    except:
                        continue
                with open(file_result, "r") as f:
                    lines = f.readlines()
                    for line in lines:
                        try:
                            v = line.strip("\n")
                            # print(v)
                            self.label.append(int(v))
                        except:
                            continue
            # split datasets
            else:
                for line in lines[start: start + len]:
                    try:
                        self.items.append([float(v) for v in line.strip("\n").split(",")[:-1]])
                        if line.strip("\n").split(",")[-1] == 'BENIGN':
                            self.label.append(0)
                        else:
                            self.label.append(1)
                    except:
                        continue

        self.items = np.array(self.items, dtype=np.float32)
        self.items = (self.items - np.mean(self.items, axis=1, keepdims=True)) / (
                np.std(self.items, axis=1, keepdims=True) + 0.00001)
        self.label = np.array(self.label)
        # print(self.label)

    def __len__(self):
        # print('item的长度是',len(self.items))
        return len(self.items)

    def __getitem__(self, idx):
        # print(self.items[idx,:],self.label[idx])
        return self.items[idx, :], self.label[idx]


class Dataset_no_label(Dataset):
    def __init__(self, input_file):
        self.items = []
        with open(input_file, "r") as f:
            lines = f.readlines()
            # for line in lines[1:10000]:  # 控制对抗样本数量
            for line in lines[1:]:  # 控制对抗样本数量
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

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return np.array(self.items[idx], dtype=np.float32)


class Dataset_G(Dataset):
    def __init__(self, input_file):
        self.items = np.load(input_file).reshape([-1,78])
        # print(self.items)


    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return np.array(self.items[idx], dtype=np.float32)