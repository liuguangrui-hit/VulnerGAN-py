import numpy as np


def load_csv(file_path, load_items=-1, max_value=1e6):
    X = []
    with open(file_path, "r") as f:
        lines = f.readlines()
        if load_items == -1:
            load_items = len(lines)
        for line in lines[:load_items]:
            try:

                # if 'inf' in line or 'nan' in line:
                #     continue
                X.append([min(float(v), max_value) for v in line.strip("\n").split(",")])
            except:
                continue
    return X


def normalize(t):
    t_mean = np.mean(t, axis=1, keepdims=True)
    t_std = np.std(t, axis=1, keepdims=True) + 0.00001
    return (t - t_mean) / t_std, t_mean, t_std
