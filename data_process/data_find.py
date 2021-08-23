import pandas as pd
import numpy as np

input_path_1 = "../data/cic_2017/data_steal/data/"
input_path_2 = "../data/cic_2017/data_steal/data_split/"
output_path_3 = "../data/cic_2017/data_steal/data_test/"
attack_list = ['Bot', 'DDoS', 'DOS', 'Patator', 'PortScan', 'Web_Attack']


for attack in attack_list:
    file_name_1 = input_path_1 + attack + ".csv"
    file_name_2 = input_path_2 + attack + ".csv"
    df1 = pd.read_csv(file_name_1, low_memory=False)
    df2 = pd.read_csv(file_name_2, low_memory=False)
    # print(df1)
    # print(df2)
    df_diff = pd.concat([df1, df2, df2]).drop_duplicates(keep=False)
    l = len(df_diff)
    print(l)
    be_file = "../data/cic_2017/data_steal/data/train_BENIGN.csv"
    df3 = pd.read_csv(be_file, low_memory=False)
    df_tr = df3.sample(n=l, replace=True)
    print(len(df_tr))
    result = pd.concat([df_diff, df_tr])
    print(result)
    file_name_3 = output_path_3 + attack + ".csv"
    result.to_csv(file_name_3, sep=',', index=False, mode='w', line_terminator='\n',
                    encoding='utf-8')

# with open(be_file, "r") as f:
#     lines = f.readlines()
#
#     for line in lines:
#         try:
#             items.append([v for v in line.strip("\n").split(",")])
#         except:
#             continue
#     # items = np.array(items)
#     np.random.shuffle(items)  ##打乱文件列表
#     cnt_test = round(len(items) * ratio_val, 0)
#     cnt_train = len(items) - cnt_test
#     train_list = []
#     test_list = []
#     for i in range(int(cnt_train)):
#         train_list.append(items[i])
#
#     for i in range(int(cnt_train), len(items)):
#         test_list.append(items[i])
#     train_df = pd.DataFrame(data=train_list)
#     test_df=pd.DataFrame(data=test_list)
#     train_df.to_csv(train_file, sep=',', header=None, index=False, mode='w', line_terminator='\n', encoding='utf-8')
#     test_df.to_csv(test_file, sep=',', header=None, index=False, mode='w', line_terminator='\n', encoding='utf-8')