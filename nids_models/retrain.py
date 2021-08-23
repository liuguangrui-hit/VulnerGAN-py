from data_process.my_dataset import Dataset_adv, Dataset, Dataset_mix
from tensorflow.keras.models import Sequential, load_model, Model
from models.save import save

import tensorflow as tf
import keras

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
keras.backend.tensorflow_backend.set_session(sess)

if __name__ == '__main__':
    # get and process data
    # data = DataProcess()
    # x_train, y_train, x_test, y_test = data.return_processed_cicids_data_binary()
    j = 0.1
    epoch = 1
    for i in range(1,10):
    #i = 3
        j += 0.1
        train_s = Dataset("../data/cic_2017/data_sets/1.0_train_set.csv",start=i*50000,len=50000)
        x_train, y_train = train_s.items, train_s.label

        test_s = Dataset("../data/cic_2017/data_sets/1.0_test_set.csv")
        x_test, y_test = test_s.items, test_s.label

        # reshape input to be [samples, timesteps, features]
        x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1])
        x_test = x_test.reshape(x_test.shape[0], 1, x_test.shape[1])

        # model_name: mlp, dnn, conv, rnn, gru, lstm
        model_name = 'epoch1_dnn'
        #j = 0.4
        model = load_model('./model_record/' + model_name + '_model.hdf5')
        model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epoch, batch_size=32, shuffle=False)
        save(model, 0, model_name)
        save(model, j, model_name)
