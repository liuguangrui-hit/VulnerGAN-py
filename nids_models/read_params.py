import os

from tensorflow.keras.models import load_model
import numpy as np
import tensorflow._api.v2.compat.v1 as tf

tf.disable_v2_behavior()

model_name = 'dnn'
model = load_model('./model_record/' + model_name + '_model.hdf5')

char_list = ["/", ":"]

# for layer in model.layers:
#     for weight in layer.get_weights():
names = [weight.name for layer in model.layers for weight in layer.weights]
weights = model.get_weights()
for name, weight in zip(names, weights):
        print("name,shape:", name, weight.shape)
        print(weight)
        # with tf.Session() as sess:
        #     sess.run(tf.global_variables_initializer())
        #     print('------------------打印出已经初始化之后的Variable的值------------------------------')
        #     print(sess.run(weight))
        #     print('----------weight的类型------------')
        #     print("type:", type(weight))
        #     # Variable转换为Tensor
        #     # Variable类型转换为tensor类型(无论是numpy转换为Tensor还是Variable转换为Tensor都可以使用tf.convert_to_tensor)
        #     data_tensor = tf.convert_to_tensor(weight)
        #     # 打印出Tensor的值（由Variable转化而来）
        #     print('------------------Variable转化为Tensor，打印出Tensor的值--------------------------')
        #     print("Tensor:",sess.run(data_tensor))
        #     # tensor转化为numpy
        #     print('-------------------tensor转换为numpy，打印出numpy的值-----------------')
        #     data_numpy = data_tensor.eval()
        #     print("numpy:",data_numpy)
        #
        #     print('---------------Variable转换为numpy（也是使用eval）--------------------')
        #     data_numpy2 = weight.eval()
        #     print("numpy_2:",data_numpy2)
        # value = np.asarray(weight.value).reshape(1, -1)
        # print(value)
        # print(value.shape)
        # name = weight.name
        for i in char_list:
            if i in name:
                name = name.replace(i, '_')
        np.savetxt('param_data/' + name + '.csv', weight, delimiter=',')
