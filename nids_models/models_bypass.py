from data_process.my_dataset import Dataset, Dataset_adv_1
from tensorflow.keras.models import load_model
import numpy as np

import tensorflow as tf
import keras

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
keras.backend.tensorflow_backend.set_session(sess)


def get_bypass(model, X_test, Y_test, model_name):
    x_test_re = x_test.reshape(x_test.shape[0], 1, x_test.shape[1])
    y_pred = model.predict(x_test_re)
    y_pred = np.array(y_pred)
    # print(y_pred)
    y_pred = [np.round(x) for x in y_pred]
    # print(y_pred)
    bypass = 0
    cnt = X_test.shape[0]
    for i in range(X_test.shape[0]):
        if Y_test[i] == 1 and y_pred[i] == 0:
            bypass += 1
        elif Y_test[i] == 0 and y_pred[i] == 1:
            bypass += 1
    print("{}_bypass: {} / {} ({})".format(model_name, bypass, cnt, bypass / cnt * 100))


if __name__ == '__main__':

    # model_name or adv_name: rnn, gru, lstm
    model_name = 'lstm'
    adv_name = 'lstm'
    # sample_size: 0.1, 1.0
    sample_size = 0.1
    test_s = Dataset_adv_1("../data/cic_2017/adver_sets/time1_"+ str(sample_size) + "_"+adv_name+"_adver_example.csv")
    x_test, y_test = test_s.items, test_s.label

    model = load_model('./model_record/epoch5_' + model_name + '_model.hdf5')
    get_bypass(model, x_test, y_test, model_name)
