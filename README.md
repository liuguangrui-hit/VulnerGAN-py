# VulnerGAN-py
A backdoor attack by vulnerability amplification on online machine learning-based network intrusion detection system



# Filename: data
## Description:
    1. 数据存储
    2. 原始CIC-IDS-2017数据集
    3. 生成的训练、测试、交叉验证集
    4. 对抗样本集

# Filename: data_process
## Description:
    1. 数据预处理
        - extract_traffic_types.py input: data/cic_2017/resource/       output: data/cic_2017/traffic_types/
        - create_datasets.py       input: data/cic_2017/traffic_types/  output: data/cic_2017/data_sets/
    2. 从训练数据集中提取异常流量（用于后续对抗样本制作）
        - create_attack_datasets.py   
		- input: data/cic_2017/data_sets/train_set.csv
        	- output: data/cic_2017/data_sets/attack_set.csv
	3. 按比例（可自定义）对抗样本划分为训练集和测试集：训练集用于投毒，测试集用于对抗样本的绕过率/制作标准验证集测模型准确率
        - split_adv_datasets.py 
		- input: data/cic_2017/adver_sets/adver_example.csv
        	- output: data/cic_2017/adver_sets/adver_train.csv & data/cic_2017/adver_sets/adver_test.csv
		
# Filename: nids_models
## Description:
    1. 模型训练
    2. 获取最后一次训练的模型可以正确/错误分类的异常样本
        - NIDS_MLP/DNN.py          
		- input: data/cic_2017/data_sets/      
		- output: nids_models/data/ & nids_models/saved_model/
    3. 测试不同数据集对模型的绕过率
        - test_MLP/DNN_bypass.py     
		- input: 生成的数据集                    
		- output: 绕过率

# Filename: GAN
## Description:
    1. 生成对抗样本
    2. GAN训练过程
    3. train_GAN.py 
	    - input:  nids_models/data/...success....npy & nids_models/data/...fail....npy
        - output: GAN/discriminator/ & GAN/generator (logs_gan 保存日志）
    4. 对抗样本生成过程
    5. generate_adver.py  
	    - input: data/cic_2017/data_sets/attack_set.csv
        - output: put: data/cic_2017/adver_sets/..._adver_example.csv
    6. find_attack.py 攻击性筛选 用于GAN训练过程中，生成的新的且保留攻击性漏洞样本，控制新样本与原input的欧式距离，生成更多可对抗样本
    7. find_weakness.py 欺骗性筛选 用于GAN训练过程中，生成的新的漏洞样本，以扩大漏洞样本库，生成更多可对抗样本

# Filename: poisoning
## Description:
    1. 投毒攻击（攻击方式可自行设计
    2. poisoning_NIDS_MLP/DNN.py
		- (非投毒) input: data/cic_2017/data_sets/ 中的训练集、测试集、交叉验证集
		- (投毒)  input: data/cic_2017/data_sets/ & data/cic_2017/adver_sets/adver_train.csv
		- output: 模型记录 poisoning/saved_models/
    3. 投毒过程记录（这里的控制台输出信息自行复制粘，保存至txt） poisoning/poisoning_record/
   
# Filename: model _steal
## Description:
    1. 生成影子模型
    2. 使用DNN对原始NIDS进行模型萃取
    3. 输入为训练样本特征和原始模型预测结果，输出为各个模型的DNN替代模型
    4. get_result.py 获取原始/替代模型的训练结果
    5. get_targets.py 获取GAN训练所需的input（nids可以正确判断的success）和target（nids误判的漏洞fail）
