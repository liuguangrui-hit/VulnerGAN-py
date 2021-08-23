# -*- coding: utf-8 -*-
#

"""
Filename: create_datasets.py
Description:
    -  Extract samples of each traffic type to generate training set,
    cross-validation set and test set

"""

import pandas as pd
import numpy as np
from datetime import datetime
import os

np.random.seed(0)


def get_typelist(df):
    """
    Extract traffic type from a pandas data frame containing IDS2017 CSV
    file with labelled traffic

    Parameter
    ---------
    df: DataFrame
        Pandas DataFrame corresponding to the content of a CSV file

    Return
    ------
    traffic_type_list: list
        List of traffic types contained in the DataFrame
    """
    traffic_type_list = df[' Label'].value_counts().index.tolist()
    return traffic_type_list


def string2index(string):
    """
    Convert a string to int so that it can be used as index in an array

    Parameter
    ---------
    string: string
        string to be converted

    Return
    ------
    index: int
        index corresponding to the string
    """
    if string == 'BENIGN':
        index = 0
    elif string == 'FTP-Patator':
        index = 1
    elif string == 'SSH-Patator':
        index = 2
    elif string == 'DoS Hulk':
        index = 3
    elif string == 'DoS GoldenEye':
        index = 4
    elif string == 'DoS slowloris':
        index = 5
    elif string == 'DoS Slowhttptest':
        index = 6
    elif string == 'Heartbleed':
        index = 7
    elif string == 'Web Attack-Brute Force':
        index = 8
    elif string == 'Web Attack-XSS':
        index = 9
    elif string == 'Web Attack-Sql Injection':
        index = 10
    elif string == 'Infiltration':
        index = 11
    elif string == 'Bot':
        index = 12
    elif string == 'PortScan':
        index = 13
    elif string == 'DDoS':
        index = 14
    else:
        print("[ERROR] Cannot convert ", string)
        index = -1
    return index


def index2string(index):
    """
    Convert an int to string

    Parameter
    ---------
    index: int
        index to be converted

    Return
    ------
    string: string
        string corresponding to the string

    """
    if index == 0:
        string = 'BENIGN'
    elif index == 1:
        string = 'FTP-Patator'
    elif index == 2:
        string = 'SSH-Patator'
    elif index == 3:
        string = 'DoS Hulk'
    elif index == 4:
        string = 'DoS GoldenEye'
    elif index == 5:
        string = 'DoS slowloris'
    elif index == 6:
        string = 'DoS Slowhttptest'
    elif index == 7:
        string = 'Heartbleed'
    elif index == 8:
        string = 'Web Attack Brute Force'
    elif index == 9:
        string = 'Web Attack XSS'
    elif index == 10:
        string = 'Web Attack Sql Injection'
    elif index == 11:
        string = 'Infiltration'
    elif index == 12:
        string = 'Bot'
    elif index == 13:
        string = 'PortScan'
    elif index == 14:
        string = 'DDoS'
    else:
        print("[ERROR] Cannot convert {}".format(index))
        string = 'Error'
    return string


def random_permutation(df_list):
    """
    Run permutations in the dataset

    Parameters
    ---------
    df_list: list
        list of pandas DataFrames, each DataFrames containing one traffic type

    Return
    ------
    reordered_df_list: array
        Resulting array of pandas DataFrames
    """
    df_list_size = len(df_list)
    reordered_df_list = df_list
    for idx in range(df_list_size):
        # Shuffle rows with a given seed to reproduce always same result
        reordered_df_list[idx] = df_list[idx].sample(frac=1, replace=False,
                                                     random_state=0)
    return reordered_df_list


def get_traffic(dataframe):
    """
    Analyze traffic of pandas data frame containing IDS2017 CSV file with
    labelled traffic

    Parameter
    ---------
    dataframe: DataFrame
        Pandas DataFrame corresponding to the content of a CSV file

    Return
    ------
    df_stats: DataFrame
        Returns a pandas data frame of one column containing amount of lines
        for each type of traffic
    """
    stats = np.zeros(15)
    stats = stats.reshape(15, 1)
    # check that all samples have been labeled
    n_samples = dataframe.shape[0]
    labels = dataframe[' Label'].value_counts()
    labels_list = get_typelist(dataframe)
    n_labels = labels.sum()
    if n_labels != n_samples:
        print("\t[INFO] missing labels: {}".format(n_samples - n_labels))
    else:
        print("\t[INFO] no missing labels")
    # write stats about traffic in an array
    idx = 0
    for lbl in labels_list:
        stats[string2index(lbl), 0] = labels[idx]
        idx += 1
    # create a data frame from the array
    df_stats = pd.DataFrame(stats,
                            index=['BENIGN',
                                   'FTP-Patator',
                                   'SSH-Patator',
                                   'DoS Hulk',
                                   'DoS GoldenEye',
                                   'DoS Slowloris',
                                   'DoS Slowhttptest',
                                   'Heartbleed',
                                   'Web Attack-Brute Force',
                                   'Web Attack-XSS',
                                   'Web Attack-SQL Injection',
                                   'Infiltration',
                                   'Bot',
                                   'PortScan',
                                   'DDoS'])
    print("\t[INFO] Analysis done")
    return df_stats


def split_dataset(df_list, df_size, randomize=False, benign_clipping=True,
                  training_percentage=60,
                  crossval_percentage=20,
                  test_percentage=20):
    """
    Split a dataset provided as an array of pandas DataFrames, each DataFrames
    containing one traffic type

    Parameter
    ---------
    df_list: array
        Array of pandas DataFrames
    randomize: boolean
        When True, data lines are permuted randomly before splitting datasets
        (default = False)
    benign_clipping: boolean
        When True, amount of traffic is set to represent 50% of overall
        traffic
    training_percentage: int
        Value between 0 and 100 corresponding to the percentage of dataset
        used to create the training set (default = 60% of dataset)
    dev_percentage: int
        Value between 0 and 100 corresponding to the percentage of dataset
        Note that values can be a float as long as percentage of dataset
        is an integer
        used to create the training set (default = 20% of dataset)
    test_percentage: int
        Value between 0 and 100 corresponding to the percentage of dataset
        used to create the training set (default = 20% of dataset)

    Return
    ------
    train_set: DataFrame
        Pandas DataFrame used as training set
    cv_set: DataFrame
        Pandas DataFrame used as cross validation set
    test_set: DataFrame
        Pandas DataFrame used as test set
    """
    print("[INFO] Splitting dataset")
    # Check percentage values
    if training_percentage + crossval_percentage + test_percentage != 100:
        print("[ERROR] Sum of percentages != 100")
        exit(-1)
    # Randomize dataset if requested
    if randomize is True:
        df_list = random_permutation(df_list)
    # Declare DataFrame to be returned
    train_set = pd.DataFrame()
    cv_set = pd.DataFrame()
    test_set = pd.DataFrame()
    # Select subset of each dataset except Benign traffic
    df_list_size = len(df_list)
    for idx in range(1, df_list_size):
        n_rows = df_list[idx].shape[0] * df_size
        n_training = int(n_rows * training_percentage / 100)
        n_crossval = int(n_rows * crossval_percentage / 100)
        n_test = int(n_rows * test_percentage / 100)
        if index2string(idx) == 'BENIGN' and benign_clipping is True:
            # Limit BENIGN traffic so that amount of BENIGN examples in attack
            # files is the same as amount of attack examples
            clipping_value = n_crossval
            training_end = clipping_value
        else:
            training_end = n_training
        crossval_end = training_end + n_crossval
        test_end = crossval_end + n_test
        train_set = train_set.append(df_list[idx][:training_end])
        cv_set = cv_set.append(df_list[idx][training_end:crossval_end])
        test_set = test_set.append(df_list[idx][crossval_end:test_end])
        print("\tdf {}".format(idx))
        print("\t# instances: {}".format(n_rows))
        print("\t# training instances: {}".format(n_training))
        print("\t# crossval instances: {}".format(n_crossval))
        print("\t# test instances: {}".format(n_test))
    # Handle specific case of Benign traffic
    n_train_attacks = train_set.shape[0]
    training_end = n_train_attacks
    n_cv_attacks = cv_set.shape[0]
    crossval_end = training_end + n_cv_attacks
    n_test_attacks = test_set.shape[0]
    test_end = crossval_end + n_test_attacks
    train_set = train_set.append(df_list[0][:training_end])
    cv_set = cv_set.append(df_list[0][training_end:crossval_end])
    test_set = test_set.append(df_list[0][crossval_end:test_end])
    # Shuffle datasets
    train_set = train_set.sample(frac=1)
    cv_set = cv_set.sample(frac=1)
    test_set = test_set.sample(frac=1)
    # Display size of each DataFrame
    print("Training set shape: {}".format(train_set.shape))
    print("Cross-val set shape: {}".format(cv_set.shape))
    print("Test set shape: {}".format(test_set.shape))
    # return resulting DataFrames
    return train_set, cv_set, test_set


def detect_non_informative_features(df_train, df_cv, df_test):
    """
    Detection of features that do not carry any information

    Parameters
    ----------
    df_train: DataFrame
        Contains training instances
    df_cv: DataFrame
        Contains crossval instances
    df_test: DataFrame
        Contains test instances

    Returns
    -------
    feature_list: list
        Contains features that can be dropped
    """
    feature_set = df_train.columns
    feature_list = []
    for i, feat in enumerate(feature_set):
        val_min_cv = df_cv[feature_set[i]].min()
        val_max_cv = df_cv[feature_set[i]].max()
        if val_min_cv == val_max_cv:
            val_min_test = df_test[feature_set[i]].min()
            val_max_test = df_test[feature_set[i]].max()
            if val_min_test == val_max_test:
                val_min_train = df_train[feature_set[i]].min()
                val_max_train = df_train[feature_set[i]].max()
                if val_min_train == val_max_train:
                    feature_list.append(feat)
                    print("Feature {} can be dropped - min = max = {}".
                          format(feat, val_min_train))
    return feature_list


def main():
    """
    Main program

    Returns
    -------
    None

    """
    # declare useful variables
    input_path = "../data/cic_2017/traffic_types/"
    pq_list = ["BENIGN.csv",
               "FTP-Patator.csv",
               "SSH-Patator.csv",
               "DoS-Hulk.csv",
               "DoS-GoldenEye.csv",
               "DoS-slowloris.csv",
               "DoS-Slowhttptest.csv",
               "Heartbleed.csv",
               "Web_Attack-Brute_Force.csv",
               "Web_Attack-XSS.csv",
               "Web_Attack-Sql_Injection.csv",
               "Infiltration.csv",
               "Bot.csv",
               "PortScan.csv",
               "DDoS.csv"]
    df_list = [pd.DataFrame(),  # benign
               pd.DataFrame(),  # ftp_patator
               pd.DataFrame(),  # ssh_patator
               pd.DataFrame(),  # dos_hulk
               pd.DataFrame(),  # dos_goldeneye
               pd.DataFrame(),  # dos_slowloris
               pd.DataFrame(),  # dos_slowhttptest
               pd.DataFrame(),  # heartbleed
               pd.DataFrame(),  # webattack_bruteforce
               pd.DataFrame(),  # webattack_xss
               pd.DataFrame(),  # webattack_sqlinjection
               pd.DataFrame(),  # infiltration
               pd.DataFrame(),  # bot
               pd.DataFrame(),  # portscan
               pd.DataFrame()]  # ddos
    output_path = "../data/cic_2017/"

    # loop over each file to load DataFrames
    for idx, filename in enumerate(pq_list):
        if filename != "Label.csv":
            print("[INFO] Reading ", filename, "...")
            # Load one file as a data frame
            df_list[idx] = pd.read_csv(input_path + filename)
            print(df_list[idx])
            # # Remove previous index column
            # drop_col = 'index'
            # df_list[idx] = df_list[idx].drop(columns=drop_col, axis=1)
            print("\t Shape: {}".format(df_list[idx].shape))
    # split dataset
    # 确定数据集提取数据的数量 df_size 1.0：全部 0.1: 10%
    # 划分训练集、交叉验证集、测试集的比例
    df_size = 0.1
    (df_train,
     df_cv,
     df_test) = split_dataset(
        df_list, df_size=df_size, randomize=True, training_percentage=40,
        crossval_percentage=40,
        test_percentage=20)
    # identify useless features
    # detect_non_informative_features(df_train, df_cv, df_test)
    # write in csv file
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    print("[INFO] {} - Writing csv files ...".format(now))
    logdir = output_path + "data_sets/"
    try:
        os.mkdir(logdir)
    except OSError as err:
        print("Creation of directory {} failed:{}".format(logdir, err))
    print(df_train)

    df_train.to_csv(logdir + str(df_size) + "_train.csv", sep=',', index=False, mode='w', line_terminator='\n',
                    encoding='utf-8')
    df_cv.to_csv(logdir + str(df_size) + "_val.csv", sep=',', index=False, mode='w', line_terminator='\n', encoding='utf-8')
    df_test.to_csv(logdir + str(df_size) + "_test.csv", sep=',', index=False, mode='w', line_terminator='\n', encoding='utf-8')


if __name__ == "__main__":
    main()
