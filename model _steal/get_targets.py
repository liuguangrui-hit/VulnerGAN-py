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


def get_test(model, X_test, Y_test,model_name,attack_name):
    correct = 0
    acc = 0
    y_pred = model.predict(X_test)
    y_pred = np.array(y_pred)
    y_pred = [np.round(x) for x in y_pred]
    y_label = np.array(Y_test)
    # for i in range(len(y_pred)):
    #     if y_pred[i][0] == y_label[i]:
    #         correct += 1
    fail_samples = []
    success_samples = []
    for i in range(X_test.shape[0]):
        if Y_test[i] == 1 and y_pred[i] == 0:
            # print(X_test[i])
            fail_samples.append(X_test[i])
        elif Y_test[i] == 1 and y_pred[i] == 1:
            success_samples.append(X_test[i])
            correct += 1
    print("fail_samples size:", len(fail_samples))
    print(fail_samples)
    fail_samples = np.array(fail_samples)
    print(fail_samples)
    target_path = 'target_record/' + model_name + '/'
    if not os.path.exists(target_path):
        os.mkdir(target_path)
    # np.save(target_path + attack_name + '_fail.npy', fail_samples)
    print("success_samples size:", len(success_samples))
    success_samples = np.array(success_samples)
    input_path = 'input_record/' + model_name + '/'
    if not os.path.exists(input_path):
        os.mkdir(input_path)
    # np.save(input_path + attack_name + '_success.npy', success_samples)
    cnt = X_test.shape[0]
    acc = correct / cnt
    print('Test set: Accuracy: {}/{} ({:.6f}%)\n'.format(correct, cnt, 100. * correct / cnt))
    return acc



if __name__ == '__main__':
    attack_list = ['Bot', 'DDoS', 'DOS', 'Patator', 'PortScan', 'Web_Attack']
    # attack_name = 'Web_Attack'
    model_name = 'RNN'
    for attack_name in attack_list:
        print(attack_name)
        test_path = "../data/cic_2017/data_steal/data/" + attack_name + ".csv"
        test_s = Dataset(test_path)
        x_test, y_test = test_s.items, test_s.label
        x_test = x_test.reshape(x_test.shape[0], 1, x_test.shape[1])
        # model_name: mlp, dnn, conv, rnn, gru, lstm
        model_p = "nids_model/" + model_name + "/" + attack_name + ".h5"
        model = load_model(model_p)
        loss, accuracy = model.evaluate(x_test, y_test, batch_size=32)
        print(accuracy)
        get_test(model, x_test, y_test,model_name,attack_name)
