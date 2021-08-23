import pandas as pd
import numpy as np

model_list = ['MLP', 'DNN', 'RNN', 'LSTM', 'GRU']
attack_list = ['Bot', 'DDoS', 'DOS', 'Patator', 'PortScan', 'Web_Attack']

ratio_train = 0.5 #训练集比例
ratio_val = 0.5 #测试集比例


for attack in attack_list:
    file = "../data/cic_2017/data_steal/data_split/" + attack + ".csv"
    items = []

    with open(file, "r") as f:
        lines = f.readlines()

        for line in lines[1:]:
            try:
                items.append([v for v in line.strip("\n").split(",")])
            except:
                continue
        # items = np.array(items)
        np.random.shuffle(items)  ##打乱文件列表
        cnt_test = round(len(items) * ratio_val, 0)
        cnt_train = len(items) - cnt_test
        train_list = []
        test_list = []
        for i in range(int(cnt_train)):
            train_list.append(items[i])

        for i in range(int(cnt_train), len(items)):
            test_list.append(items[i])
        train_df = pd.DataFrame(data=train_list)
        test_df=pd.DataFrame(data=test_list)

        train_file = "../data/cic_2017/data_steal/data_train/" + attack + "_train.csv"
        test_file = "../data/cic_2017/data_steal/data_test/" + attack + "_test.csv"
        train_df.to_csv(train_file, sep=',', index=False, mode='w', line_terminator='\n', encoding='utf-8')
        test_df.to_csv(test_file, sep=',', index=False, mode='w', line_terminator='\n', encoding='utf-8')