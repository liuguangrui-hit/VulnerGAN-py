import joblib

from nids_models.NIDS import NIDS
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from data_process.tools import load_csv, normalize
import numpy as np


class Dataset():
    def __init__(self, input_files, train, items, do_normalize=True):
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


# seven models
__all__ = {
    "gnb": GaussianNB,  # 朴素贝叶斯
    "knn": KNeighborsClassifier,  # k近邻
    "gbc": GradientBoostingClassifier,  # 梯度提升树
    "rfc": RandomForestClassifier,  # 随机森林
    "lr": LinearRegression,  # 线性回归
    "svc": SVC,  # 支持向量机
    "dtc": DecisionTreeClassifier  # 决策树
}

if __name__ == "__main__":
    model_name = "knn"
    model = __all__[model_name]()

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

    # start NIDS training
    if not reuse_model and is_train:
        nids = NIDS(model,
                    ["../data/cic_2017/data_sets/0.1_train_set.csv", "../data/cic_2017/data_sets/0.1_test_set.csv"],
                    [True, False], [3000, 3000])
        nids.train()
        nids.get_metrics()
        nids.train()
        nids.save(f"saved_model/{model_name}.m")
        nids.get_metrics()
        # nids.predict(f"data/target_{model_name}.csv")
        nids.get_fail_samples(f"data/fail_{model_name}.npy")
        nids.get_success_samples(f"data/success_{model_name}.npy")

    # continue NIDS training
    elif reuse_model and is_train:
        # ds = Dataset(["../data/cic_2017/data_sets/0.1_train_set.csv",
        # "../data/cic_2017/data_sets/0.1_test_set.csv"], [True, False], [3000, 3000])
        model_path = f"saved_model/{model_name}.m"
        clf = joblib.load(model_path)
        nids = NIDS(clf,
                    ["../data/cic_2017/data_sets/0.1_train_set.csv", "../data/cic_2017/data_sets/0.1_test_set.csv"],
                    [True, False], [3000, 3000])
        nids.train()
        nids.save(f"saved_model/{model_name}.m")
        nids.get_metrics()

    # test
    elif reuse_model and not is_train:
        model_path = f"saved_model/{model_name}.m"
        clf = joblib.load(model_path)
        nids = NIDS(clf,
                    ["../data/cic_2017/data_sets/0.1_train_set.csv", "../data/cic_2017/data_sets/0.1_test_set.csv"],
                    [True, False], [3000, 3000])
        nids.get_metrics()
