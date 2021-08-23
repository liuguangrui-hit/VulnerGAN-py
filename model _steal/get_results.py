import os
import shutil

from data_process.my_dataset import Dataset
from tensorflow.keras.models import load_model
import numpy as np

import tensorflow as tf
import keras

config=tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess=tf.compat.v1.Session(config=config)
keras.backend.tensorflow_backend.set_session(sess)


def get_test(model, X_test, Y_test, file_name):
    correct = 0
    acc = 0
    # x_test_re = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
    y_pred = model.predict(X_test)
    y_pred = np.array(y_pred)
    y_pred = [np.round(x) for x in y_pred]
    with open(file_name, "a") as f:
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



if __name__ == '__main__':
    model_list = ['RNN', 'LSTM', 'GRU']
    model_name = 'GRU'
    attack_list = ['Bot', 'DDoS', 'DOS', 'Patator', 'PortScan', 'Web_Attack']
    attack_name = 'Bot'
    for model_n in model_list:
        print(model_n)
        file_path = "data_record/" + model_n + "/"
        if not os.path.exists(file_path):
            os.mkdir(file_path)
        # model_name: rnn, gru, lstm
        for attack in attack_list:
            print(attack)
            test_s = Dataset("../data/cic_2017/data_steal/data_train/" + attack + "_train.csv")
            x_test, y_test = test_s.items, test_s.label
            x_test = x_test.reshape(x_test.shape[0], 1, x_test.shape[1])
            file_saved = file_path + attack + ".csv"
            if os.path.exists(file_saved):
                os.unlink(file_saved)
            model_p = "nids_model/" + model_n + "/" + attack + ".h5"
            model = load_model(model_p)
            get_test(model, x_test, y_test, file_saved)
