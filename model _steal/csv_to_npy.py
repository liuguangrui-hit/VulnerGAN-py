import numpy as np
import pandas as pd


# 先用pandas读入csv
data = np.array(pd.read_csv("input_target_csv/MLP/Bot_fail.csv")).tolist()
new_list = []
for d in data:
    # print(d[:-1])
    new_list.append(d[:-1])

data = np.array(new_list, dtype=np.float32)
print(data)
# 再使用numpy保存为npy
np.save("input_target_csv/MLP/Bot_fail.npy", data)

target_items = np.load('input_target_csv/MLP/Bot_fail.npy')
print(target_items)
print(target_items.shape)
# target_items=target_items.reshape([-1,78])
# print(target_items.shape)
# print(len(target_items))
#
# target_items = np.load('target_record/MLP/Bot_fail.npy')
# print(target_items.shape)
# target_items=target_items.reshape([-1,78])
# print(target_items.shape)
# print(len(target_items))