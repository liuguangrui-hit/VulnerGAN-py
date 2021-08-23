import numpy as np


def load_csv(file_path, load_items=-1, max_value=1e6):
    X = []
    label = []
    with open(file_path, "r") as f:
        lines = f.readlines()
        if load_items == -1:
            load_items = len(lines)
        for line in lines[:load_items]:
            try:

                # if 'inf' in line or 'nan' in line:
                #     continue
                X.append([float(v) for v in line.strip("\n").split(",")[:-1]])
                label.append(numeralization(line.strip("\n").split(",")[-1]))
            except:
                continue
    # print(X, label)
    # print(len(X), len(label))
    return X, label


def numeralization(label_str):
    if label_str == 'BENIGN':
        return 0
    else:
        return 1


def normalize(t):
    t_mean = np.mean(t, axis=1, keepdims=True)
    t_std = np.std(t, axis=1, keepdims=True) + 0.00001
    return (t - t_mean) / t_std, t_mean, t_std


if __name__ == '__main__':
    tmp_X, tmp_Y = load_csv("../data/cic_2017/data_sets/train_set.csv", 3000)
