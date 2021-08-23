import os

from data_process.my_dataset import Dataset_adv, Dataset, Dataset_mix,Dataset_adv_1
from tensorflow.keras.models import Sequential, load_model, Model
from poisoning.save_model import save_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Activation, Dropout, SimpleRNN
import numpy as np
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader

import tensorflow as tf
import keras
config=tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess=tf.compat.v1.Session(config=config)
keras.backend.tensorflow_backend.set_session(sess)

class my_RNN():
    def __init__(self, x_train):
        self.model = Sequential()
        self.model.add(SimpleRNN(120, input_shape=(x_train.shape[1], x_train.shape[2]), return_sequences=True))
        self.model.add(Dropout(0.2))

        self.model.add(SimpleRNN(120, return_sequences=True))
        self.model.add(Dropout(0.2))

        self.model.add(SimpleRNN(120, return_sequences=False))
        self.model.add(Dropout(0.2))

        # binary
        self.model.add(Dense(1))
        self.model.add(Activation('sigmoid'))

        # multiclass
        # model.add(Dense(5))
        # model.add(Activation('softmax'))

        self.model.summary()

        # optimizer
        adam = Adam(lr=0.0001)

        # binary
        self.model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

def get_test(model, X_test, Y_test):
    correct = 0
    acc = 0
    # x_test_re = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
    y_pred = model.predict(X_test)
    y_pred = np.array(y_pred)
    y_pred = [np.round(x) for x in y_pred]

    for i in range(X_test.shape[0]):
        if Y_test[i] == 1 and y_pred[i] == 1:
            correct += 1
        if Y_test[i] == 0 and y_pred[i] == 0:
            correct += 1
    cnt = X_test.shape[0]
    acc = correct / cnt
    print('Test set: Accuracy: {}/{} ({:.6f}%)\n'.format(correct, cnt, 100. * correct / cnt))
    return acc


epoch = 10


if __name__ == '__main__':

    # get and process data
    # data = DataProcess()
    # x_train, y_train, x_test, y_test, x_test_21, y_test_21 = data.return_processed_data_multiclass()
    # x_train, y_train, x_test, y_test = data.return_processed_cicids_data_binary()

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
    attack_list = ['DDoS', 'Web_Attack']
    attack_name = 'Bot'
    model_list = ['MLP', 'DNN', 'RNN', 'LSTM', 'GRU']
    model_name1 = 'MLP'
    model_name = 'RNN'
    test_list = ['cheat_test', 'attack_test']
    test_name = 'cheat_test'
    m = 1

    model_path = "model_record/" + test_name + "/" + model_name + "/" + attack_name + "/"

    adv_path = "../data/cic_2017/adver_sets/" + test_name + "/" + model_name1 + "/" + attack_name + "/"
    test_s = Dataset_adv_1(adv_path + "adver_" + str(m) + "_test.csv")
    x_test, y_test = test_s.items, test_s.label
    # reshape input to be [samples, timesteps, features]
    x_test = x_test.reshape(x_test.shape[0], 1, x_test.shape[1])

    dataset_n_len = 20000
    dataset_a_len = 50

    # start training
    if not reuse_model and is_train:
        for i in range(epoch):
            print("epoch:", i + 1)
            if i == 0:
                train_s = Dataset("../data/cic_2017/data_steal/data_split/"+ attack_name + "_train.csv", start=i * dataset_n_len,
                                  len=dataset_n_len)
                x_train, y_train = train_s.items, train_s.label
                # print(x_train.shape)
                x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1])
                # print(x_train.shape)
                model = my_RNN(x_train).model
                for k in range(5):
                    # compute_mse(i,model,x_train,y_train)
                    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=1, batch_size=32)
                save_model(model, 0, model_name, model_path)
                save_model(model, i+1, model_name, model_path)
            elif (i > 0 and i <= 4):
                train_s = Dataset("../data/cic_2017/data_steal/data_split/"+ attack_name + "_train.csv", start=i * dataset_n_len,
                                  len=dataset_n_len)
                x_train, y_train = train_s.items, train_s.label
                x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1])
                model = load_model(model_path + "NIDS_" + model_name + ".hdf5")

                for k in range(5):
                    # compute_mse(i,model,x_train,y_train)
                    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=1, batch_size=32)

                save_model(model, i+1, model_name, model_path)
                save_model(model, 0, model_name, model_path)
            else:
                train_s = Dataset_mix("../data/cic_2017/data_steal/data_split/"+ attack_name + "_train.csv",
                                "../data/cic_2017/adver_sets/" + test_name + "/" + model_name + "/" + attack_name + "/" + "adver_" + str(k) + "_train.csv",
                                      p_start=(i) * dataset_n_len, p_len=dataset_n_len, n_start=(i) * dataset_a_len,
                                      n_len=dataset_a_len)
                x_train, y_train = train_s.items, train_s.label
                x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1])
                model = load_model(model_path + "NIDS_" + model_name + ".hdf5")

                for k in range(5):
                    # compute_mse(i,model,x_train,y_train)
                    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=1, batch_size=32)
                save_model(model, i + 1, model_name, model_path)
                save_model(model, 0, model_name, model_path)

    elif reuse_model and is_train:
        # model_name: mlp, dnn, conv, rnn, gru, lstm

        for attack_name in attack_list:
            print(attack_name)
            model = load_model('nids_source/epoch5_RNN_model.hdf5')
            adv_path = "../data/cic_2017/adver_sets/" + test_name + "/" + model_name + "/" + attack_name + "/"
            test_s = Dataset_adv_1(adv_path + "adver_" + str(m) + "_test.csv")
            x_test, y_test = test_s.items, test_s.label
            # reshape input to be [samples, timesteps, features]
            x_test = x_test.reshape(x_test.shape[0], 1, x_test.shape[1])
            model_path = "model_record/" + test_name + "/" + model_name + "/" + attack_name + "/"
            for j in range(5):
                train_s = Dataset_mix("../data/cic_2017/data_sets/1.0_train_set.csv",
                                      "../data/cic_2017/adver_sets/" + test_name + "/" + model_name + "/" + attack_name + "/" + "adver_" + str(
                                          m) + "_train.csv",
                                      p_start=(j) * dataset_n_len, p_len=dataset_n_len, n_start=(j) * dataset_a_len,
                                      n_len=dataset_a_len)
                x_train, y_train = train_s.items, train_s.label
                x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1])

                acc = get_test(model, x_test, y_test)
                model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=5, batch_size=32, shuffle=False)
                acc = get_test(model, x_test, y_test)
                save_model(model, 0, model_name)
                save_model(model, j + 1, model_name, model_path)

    else:
        c = input("Enter k: ")
        i = int(c)
        model = load_model(model_path + "NIDS_" + model_name + ".hdf5")
        acc_min = get_test(model, x_test, y_test)
