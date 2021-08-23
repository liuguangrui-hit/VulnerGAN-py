import pandas as pd

attack_list = ['Bot', 'DDoS', 'DOS', 'Patator', 'PortScan', 'Web_Attack']
attack_name = 'DDoS'
model_list = ['MLP', 'DNN', 'RNN', 'LSTM', 'GRU']
model_name = 'MLP'

for attack_name in attack_list:
    input_path = "../data/cic_2017/data_steal/data_split/"
    input_file = input_path + attack_name + ".csv"

    df = pd.DataFrame()
    df = pd.read_csv(input_file).sample(frac=1)
    # print(df)
    df.to_csv(input_path  + attack_name + "_train.csv", sep=',', index=False, mode='w', line_terminator='\n',
                        encoding='utf-8')