# Filename: data
## Description: 存储数据
    - success.npy NIDS判断正确的异常样本
    - fail.npy NIDS判断错误的异常样本
    - epoch=1 使用完整训练数据进行在线训练 每批样本完整训练1次
    - epoch=5 使用完整训练数据进行在线训练 每批样本完整训练5次
    - .csv 模型分类结果（标签记录

# Filename: saved_model
## Description: 存储训练后的模型

# Filename: .txt
## Description: 训练过程记录

# Filename: NIDS_MLP.py / NIDS_DNN.py
## Description:
    - MLP/DNN 模型训练，过程记录，结果保存

# Filename: test_MLP/DNN_bypass.py
## Description:
    - 测试对抗样本对MLP/DNN的绕过率

# Filename: train_NIDS.py
## Description:
    - 其他机器学习模型训练过程
    -
    "gnb": GaussianNB,  # 朴素贝叶斯
    "knn": KNeighborsClassifier,  # k近邻
    "gbc": GradientBoostingClassifier,  # 梯度提升树
    "rfc": RandomForestClassifier,  # 随机森林
    "lr": LinearRegression,  # 线性回归
    "svc": SVC,  # 支持向量机
    "dtc": DecisionTreeClassifier  # 决策树

# Filename: test_bypass.py
## Description:
    - 测试对抗样本对其他模型的绕过率
