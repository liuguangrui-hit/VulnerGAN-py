# -*- coding: utf-8 -*-
#

"""
Filename: extract_traffic_types.py
Description:
    - Read all CSV file containing labels
    - Group by traffic type (label)
    - Generate one CSV file (incl. label) per traffic type
"""

import pandas as pd
from datetime import datetime
import numpy as np


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
        string = 'Web Attack-Brute Force'
    elif index == 9:
        string = 'Web Attack-XSS'
    elif index == 10:
        string = 'Web Attack-Sql Injection'
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


def get_dataframe_ofType(df, traffic_type):
    """
    Analyze traffic distribution of pandas data frame containing IDS2017 CSV
    file with labelled traffic

    Parameter
    ---------
    df: DataFrame
        Pandas DataFrame corresponding to the content of a CSV file
    traffic_type: string
        name corresponding to traffic type

    Return
    ------
    req_df: DataFrame
        Pandas DataFrame containing only the requested traffic type
    """
    req_df = df.loc[df[' Label'] == traffic_type]
    # don't keep original indexes
    req_df = req_df.reset_index(drop=True)
    return req_df


def remove_empty_lines(df):
    """
    Remove empty lines imported from csv files into Pandas DataFrame as NaN.
    For a fast processing, only FlowID is checked. If NaN, then the line is
    dropped.

    Parameters
    ----------
    df: DataFrame
        Pandas DataFrame to be inspected

    Returns
    -------
    df_clean: DataFrame
        Pandas DataFrame after clean-up
    """
    df.replace([''], np.nan, inplace=True)
    df_clean = df.dropna(subset=['Flow ID'], inplace=False)
    n_removed = df.shape[0] - df_clean.shape[0]
    if n_removed != 0:
        print("[INFO] Empty lines removed: {}".
              format(df.shape[0] - df_clean.shape[0]))
    return df_clean


def detect_drop_outliers(df):
    """
    Detect and drop NaN rows of a DataFrame

    Parameters
    ----------
    df: DataFrame
        pandas DataFrame containing data

    Returns
    -------
    clean_df: DataFrame
        pandas DataFrame without rows containing NaN
    """

    clean_df = df
    lbl_list = df.columns.values
    lbl_idx, = np.where(lbl_list == 'Flow Bytes/s')
    # print("Flow Bytes/s:", lbl_idx)
    # print(clean_df['Flow Bytes/s'].dtype)
    clean_df['Flow Bytes/s'] = np.array(
        clean_df.iloc[:, lbl_idx]).astype(np.float64)
    lbl_idx, = np.where(lbl_list == ' Flow Packets/s')
    # print(" Flow Packets/s:", lbl_idx)
    # print(clean_df[' Flow Packets/s'].dtype)
    clean_df[' Flow Packets/s'] = np.array(
        clean_df.iloc[:, lbl_idx]).astype(np.float64)
    df.replace(np.inf, np.nan,
               inplace=True)
    null_columns = clean_df.columns[clean_df.isna().any()]
    nan_cnt = clean_df[null_columns].isnull().sum()
    if nan_cnt.empty is False:
        print("\t\tNaN detected and dropped: ")
        print(nan_cnt)
        clean_df = clean_df.dropna(axis=0)
        print("\t\tPrev shape: {} - New shape: {}".format(df.shape,
                                                          clean_df.shape))
    return clean_df


def main():
    # declare useful variables
    input_path = "../data/cic_2017/resource/"
    filelist = ("Monday-WorkingHours.pcap_ISCX.csv",
                "Tuesday-WorkingHours.pcap_ISCX.csv",
                "Wednesday-workingHours.pcap_ISCX.csv",
                "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
                "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
                "Friday-WorkingHours-Morning.pcap_ISCX.csv",
                "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
                "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")
    output_path = "../data/cic_2017/traffic_types/"
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
    dflist = [pd.DataFrame(),  # benign
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
    # loop over each file
    for filename in filelist:
        t = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print("[INFO] {} - Extracting from {}".format(t, filename))
        # Load one file as a data frame
        df = pd.read_csv(input_path + filename,
                         encoding="utf-8", low_memory=False)
        print(df)
        print("\tShape: {}".format(df.shape))
        typelist = get_typelist(df)
        if ' Label' in typelist:
            idx_to_delete = [i for i, x in enumerate(typelist) if x == ' Label']
            typelist = np.delete(typelist, idx_to_delete)
        # df = remove_empty_lines(df)
        for type_l in typelist:
            print("\tAnalyzing {} ...".format(type_l))
            dflist[string2index(type_l)] = dflist[string2index(type_l)].append(
                get_dataframe_ofType(df, type_l), sort=False)
            dflist[string2index(type_l)] = detect_drop_outliers(
                dflist[string2index(type_l)])

    # create one parquet file per traffic type
    t = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("[INFO] {} - Writing csv files ...".format(t))
    for idx in range(len(dflist)):
        if not dflist[idx].empty:
            t = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print("[INFO] {} - {} shape: {}".format(t,
                                                    pq_list[idx],
                                                    dflist[idx].shape))
            dflist[idx].to_csv(output_path + pq_list[idx], sep=',', index=False, mode='w', line_terminator='\n',
                               encoding='utf-8')


if __name__ == "__main__":
    main()
