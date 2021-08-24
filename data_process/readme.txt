# Filename: extract_traffic_types.py
## Description:
    - 读取数据原始.csv文件
    - 去除包含NAN or INF值的记录
    - 按照标签类型分组
    - 按照流量类型分类生成新的.csv文件

# Filename: create_datasets.py
## Description:
    - 读取按照流量类型生成的.csv文件
    - 指定提取的记录数目
    - 指定训练集、交叉验证集、测试集的比例
    - 生成相应数据集

# Filename: tools.py
## Description:
    - 自定义.csv文件读取方式
    - 标签二分类
    - 数据标准化

# Filename: my_dataset.py
## Description:
    - 用于后续实验读取相应数据集

# Filename: create_attack_datasets.py
## Description:
    - 提取所有攻击类型流量（除label为benign外所有）

# Filename: split_adv_datasets.py
## Description:
    - 按照比例划分对抗样本训练集和测试集
    - 训练集用于投毒
    - 测试集用于制作标准验证集 or 检验漏洞放大效果
