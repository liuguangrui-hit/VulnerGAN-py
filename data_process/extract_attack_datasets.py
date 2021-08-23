import pandas as pd
import numpy as np

input_path = "../data/cic_2017/data_sets/"
filename = "0.1_train_set.csv"

file = input_path + filename
items = []

output_file = "../data/cic_2017/data_sets/0.1_attack_n.csv"

with open(file, "r") as f:
    lines = f.readlines()[1:]

    for line in lines:
        try:
            if line.strip("\n").split(",")[-1] != 'BENIGN':
                items.append([v for v in line.strip("\n").split(",")[:-1]])
        except:
            continue
    items = np.array(items, dtype=np.float32)
    items = (items - np.mean(items, axis=1, keepdims=True)) / (
            np.std(items, axis=1, keepdims=True) + 0.00001)
    test=pd.DataFrame(data=items)
    test.to_csv(output_file, sep=',', header=None, index=False, mode='w', line_terminator='\n', encoding='utf-8')