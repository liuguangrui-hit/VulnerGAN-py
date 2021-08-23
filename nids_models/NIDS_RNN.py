import datetime
import os
import shutil
import sys
import traceback
sys.path.append(os.path.dirname(sys.path[0]))
from data_process.my_dataset import Dataset_adv, Dataset, Dataset_mix, Dataset_adv_1,Dataset_shadow
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Activation, Dropout, SimpleRNN
import numpy as np
from poisoning.save_model import save_model

import tensorflow as tf
import keras

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
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

def get_test_result(model, X_test, Y_test):
    correct = 0
    acc = 0
    # x_test_re = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
    y_pred = model.predict(X_test)
    y_pred = np.array(y_pred)
    y_pred = [np.round(x) for x in y_pred]
    file_path = "data_record/0.1_NIDS_RNN_result.csv"
    if os.path.exists(file_path):
        os.unlink(file_path)
        # os.mkdir(file_path)
    with open(file_path, "a") as f:
        items = np.array(y_pred)
        np.savetxt(f, items, fmt='%d', delimiter=',')
    for i in range(X_test.shape[0]):
        if Y_test[i] == 1 and y_pred[i] == 1:
            correct += 1
        if Y_test[i] == 0 and y_pred[i] == 0:
            correct += 1
    cnt = X_test.shape[0]
    acc = correct / cnt
    print('Test set: Accuracy: {}/{} ({:.6f}%)\n'.format(correct, cnt, 100. * correct / cnt))
    return acc


# hyper-parameter
epoch = 1
model_path = "model_record/GRU_RNN/"
model_name = 'NIDS_GRU_RNN'

if __name__ == '__main__':
    reuse_model = False
    is_train = True
    loop_exit = False
    while not loop_exit:
        print("----------- Welcome to NIDS Poison Detector! -----------")
        print("Menu:")
        print("\t1: start NIDS training")
        print("\t2: NIDS test")
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


    test_s = Dataset("../data/cic_2017/data_sets/1.0_test.csv")
    x_test, y_test = test_s.items, test_s.label
    x_test = x_test.reshape(x_test.shape[0], 1, x_test.shape[1])
    # reshape input to be [samples, timesteps, features]
    # x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1])

    val_s = Dataset("../data/cic_2017/data_sets/1.0_test.csv")
    x_val, y_val = val_s.items, val_s.label
    x_val = x_val.reshape(x_val.shape[0], 1, x_val.shape[1])


    # dataset_n_len = 20000
    # dataset_a_len = 5

    # start training
    if not reuse_model and is_train:
        # 清空所有model记录
        if os.path.exists(model_path):
            shutil.rmtree(model_path)
            os.mkdir(model_path)
        if not os.path.exists(model_path):
            # shutil.rmtree(model_path)
            os.mkdir(model_path)
        # train_s = Dataset("../data/cic_2017/data_sets/1.0_train.csv")
        train_s = Dataset_shadow("../data/cic_2017/data_sets/0.1_val.csv", "data_record/0.1_NIDS_GRU_result.csv")
        x_train, y_train = train_s.items, train_s.label
        # print(x_train.shape)
        # print(y_train.shape)
        x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1])
        # print(x_train.shape)
        model = my_RNN(x_train).model

        i = 0
        while i < epoch:
            print('----------- epoch: %d -----------' % (i + 1))

            model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=1, batch_size=32)
            val = get_test(model, x_val, y_val)
            save_model(model, i+1, model_name,model_path)
            i += 1

        save_model(model, -1, model_name)

        print('----------- Model training has been completed! -----------\n\n')


    elif reuse_model and is_train:
        test_s = Dataset("../data/cic_2017/data_sets/0.1_val.csv")
        x_test, y_test = test_s.items, test_s.label
        x_test = x_test.reshape(x_test.shape[0], 1, x_test.shape[1])
        model_name = 'NIDS_RNN'
        model_p = 'model_record/RNN/1_' + model_name + ".hdf5"
        if os.path.exists(model_p):
            model = load_model(model_p)
            acc_min = get_test_result(model, x_test, y_test)
        else:
            print('No saved model, try start NIDS training！')


    # test
    elif reuse_model and not is_train:
        test_s = Dataset("../data/cic_2017/data_sets/1.0_test_set.csv")
        x_test, y_test = test_s.items, test_s.label
        x_test = x_test.reshape(x_test.shape[0], 1, x_test.shape[1])
        model_p = 'model_record/RNN/1_' + model_name + ".hdf5"
        if os.path.exists(model_p):
            model = load_model(model_p)
            acc_min = get_test(model, x_test, y_test)
        else:
            print('No saved model, try start NIDS training！')
