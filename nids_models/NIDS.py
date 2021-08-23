from data_process.tools import load_csv, normalize
import numpy as np
import joblib
import os
import pickle

from sklearn.metrics import accuracy_score
from sklearn import metrics


# nids = NIDS(model, ["../data/cic_2017/data_sets/train_set.csv", "../data/cic_2017/data_sets/test_set.csv"], [True, False], [3000, 3000])

class NIDS():
    def __init__(self, model, input_files, train, items, do_normalize=True):
        self.X_train = []
        self.Y_train = []
        self.X_test = []
        self.Y_test = []
        self.X_std = None
        self.X_mean = None
        self.model = model
        # train == True 加入训练集 else 加入测试集
        for input_file, train, item in zip(input_files, train, items):
            if train:
                tmp_X, tmp_Y = load_csv(input_file, item)
                # tmp_Y = [label] * len(tmp_X)
                self.X_train += tmp_X
                self.Y_train += tmp_Y
            else:
                tmp_X, tmp_Y = load_csv(input_file, item)
                self.X_test += tmp_X
                self.Y_test += tmp_Y
        self.X_train = np.array(self.X_train, dtype='float32')
        self.X_test = np.array(self.X_test, dtype='float32')
        if do_normalize:
            self.X_train, self.X_mean, self.X_std = normalize(self.X_train)
            self.X_test, _, _ = normalize(self.X_test)

    def train(self):
        self.model.fit(self.X_train, self.Y_train)

    def save(self, model_path):
        if not os.path.exists(os.path.dirname(model_path)):
            os.makedirs(os.path.dirname(model_path))
        with open(model_path, "wb") as f:
            joblib.dump(self.model, f)
            # pickle.dump({
            #     "model": self.model, "std": self.X_mean, "mean": self.X_std
            # }, f)

    def predict(self, output_file):
        if not os.path.exists(os.path.dirname(output_file)):
            os.makedirs(os.path.dirname(output_file))
        predict = self.model.predict(self.X_test)
        with open(output_file, "w") as f:
            for i in range(self.X_test.shape[0]):
                f.write(str(predict[i]) + "\n")

    def get_fail_samples(self, output_file):
        predict = self.model.predict(self.X_test)
        fail_samples = []
        for i in range(self.X_test.shape[0]):
            if self.Y_test[i] == 1 and predict[i] == 0:
                fail_samples.append(self.X_test[i])
        fail_samples = np.array(fail_samples)
        np.save(output_file, fail_samples)

    def get_success_samples(self, output_file):
        predict = self.model.predict(self.X_test)
        success_samples = []
        for i in range(self.X_test.shape[0]):
            if self.Y_test[i] == 1 and predict[i] == 1:
                success_samples.append(self.X_test[i])
        success_samples = np.array(success_samples)
        np.save(output_file, success_samples)

    def _predict(self, x):
        return self.model.predict(x)

    def get_metrics(self):
        y_pred = self.model.predict(self.X_test)
        acc = metrics.precision_score(self.Y_test, y_pred, average='macro')  # 宏平均，精确率
        re_call = metrics.recall_score(self.Y_test, y_pred, average='macro')
        F1_score = metrics.f1_score(self.Y_test, y_pred, average='weighted')
        print("acc:",acc)
        print("re_call:", re_call)
        print("F1_score:", F1_score)
        return  acc, re_call, F1_score
