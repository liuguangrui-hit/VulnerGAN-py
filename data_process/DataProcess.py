import os
import io
import numpy as np
from random import shuffle
from sklearn.model_selection import train_test_split


class DataProcess:
    _DATA_FOLDER = "data"

    @classmethod
    def get_current_working_directory(self):
        """
        get current directory
        """
        return os.getcwd()

    @classmethod
    def read_file_lines(self, dataset, filename):
        """
        read all lines of file with file name, not full path
        """
        filepath = os.path.join(
            self.get_current_working_directory(), 'data', dataset, filename)
        with open(filepath, 'r', encoding='utf-8') as content:
            return content.readlines()

    @classmethod
    def extract_features(self, a_line):
        """
        extract features based on comma (,), return an np.array
        """
        return [x.strip() for x in a_line.split(',')]

    @classmethod
    def numericalize_feature_cicids(self, feature):
        # make all values np.float64
        feature = [np.float64(-1) if (x == "Infinity" or x == "NaN") else np.float64(x) for x in feature]

        return np.array(feature)

    @classmethod
    def numericalize_result_cicids(self, reslut, attack, attack_dict):
        res = list()
        res[0:0] = attack[attack_dict[reslut]]
        # make all values np.float64
        res = [np.float64(x) for x in res]
        return np.array(res)

    @classmethod
    def normalize_value(self, value, min, max):
        value = np.float64(value)
        min = np.float64(min)
        max = np.float64(max)

        if min == np.float64(0) and max == np.float64(0):
            return np.float64(0)
        result = np.float64((value - min) / (max - min))
        return result

    @classmethod
    def cicids_process_data_binary(self):
        """
        read from data folder and return a list
        """
        normal_limit = 540000
        normal_count = 0

        train_data_1 = self.read_file_lines('cicids', 'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv')
        train_data_1.pop(0)
        shuffle(train_data_1)

        train_data_2 = self.read_file_lines('cicids', 'Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv')
        train_data_2.pop(0)
        shuffle(train_data_2)

        train_data_3 = self.read_file_lines('cicids', 'Friday-WorkingHours-Morning.pcap_ISCX.csv')
        train_data_3.pop(0)
        shuffle(train_data_3)

        # train_data_4 = self.read_file_lines('cicids', 'Monday-WorkingHours.pcap_ISCX.csv')
        # train_data_4.pop(0)
        # shuffle(train_data_4)
        # shuffle(train_data_4)

        train_data_5 = self.read_file_lines('cicids', 'Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv')
        train_data_5.pop(0)
        shuffle(train_data_5)

        train_data_6 = self.read_file_lines('cicids', 'Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv')
        train_data_6.pop(0)
        shuffle(train_data_6)

        train_data_7 = self.read_file_lines('cicids', 'Tuesday-WorkingHours.pcap_ISCX.csv')
        train_data_7.pop(0)
        shuffle(train_data_7)

        train_data_8 = self.read_file_lines('cicids', 'Wednesday-workingHours.pcap_ISCX.csv')
        train_data_8.pop(0)
        shuffle(train_data_8)

        # train_data = train_data_1 + train_data_2+ train_data_3+ train_data_4+ train_data_5+ train_data_6+ train_data_7+ train_data_8
        train_data = train_data_1 + train_data_2 + train_data_3 + train_data_5 + train_data_6 + train_data_7 + train_data_8

        shuffle(train_data)

        # extract data and shuffle it
        raw_train_data_features_extra = [self.extract_features(x) for x in train_data]

        # limit normal data
        raw_train_data_features = []
        for i in range(0, len(raw_train_data_features_extra)):
            if 'BENIGN' in raw_train_data_features_extra[i][-1]:
                normal_count = normal_count + 1
                if normal_limit >= normal_count:
                    raw_train_data_features.append(raw_train_data_features_extra[i])
            else:
                if len(raw_train_data_features_extra[i][-1]) > 0:
                    raw_train_data_features.append(raw_train_data_features_extra[i])

        shuffle(raw_train_data_features)

        # train data: put index 0 to 78 in data and 79  into result
        raw_train_data_results = [x[-1] for x in raw_train_data_features]
        raw_train_data_features = [x[0:-1] for x in raw_train_data_features]

        # stage 1 : numericalization
        # 1.1 extract all protocol_types, services and flags
        attack = dict()
        attack_dict = {
            'BENIGN': 'BENIGN',
            'DDoS': 'ATTACK',
            'PortScan': 'ATTACK',
            'Infiltration': 'ATTACK',
            'Web Attack-Brute Force': 'ATTACK',
            'Web Attack-XSS': 'ATTACK',
            'Web Attack-Sql Injection': 'ATTACK',
            'Bot': 'ATTACK',
            'FTP-Patator': 'ATTACK',
            'SSH-Patator': 'ATTACK',
            'DoS slowloris': 'ATTACK',
            'DoS Slowhttptest': 'ATTACK',
            'DoS Hulk': 'ATTACK',
            'DoS GoldenEye': 'ATTACK',
            'Heartbleed': 'ATTACK'
        }

        attack['BENIGN'] = [int(0)]
        attack['ATTACK'] = [int(1)]

        # train data

        numericalized_train_data_features = [self.numericalize_feature_cicids(x) for x in raw_train_data_features]
        normalized_train_data_features = np.array(numericalized_train_data_features)

        numericalized_train_data_results = [self.numericalize_result_cicids(x, attack, attack_dict) for x in
                                            raw_train_data_results]
        normalized_train_data_results = np.array(numericalized_train_data_results)

        # stage 2: normalization --> x = (x - MIN) / (MAX - MIN) --> based on columns

        # train data
        ymin_train = np.amin(normalized_train_data_features, axis=0)
        ymax_train = np.amax(normalized_train_data_features, axis=0)

        # normalize train
        for x in range(0, normalized_train_data_features.shape[0]):
            for y in range(0, normalized_train_data_features.shape[1]):
                normalized_train_data_features[x][y] = self.normalize_value(
                    normalized_train_data_features[x][y], ymin_train[y], ymax_train[y])

        train_data_features, test_data_features, train_data_results, test_data_results = train_test_split(
            normalized_train_data_features, normalized_train_data_results, test_size=0.35)

        mul_cicids = os.path.join(
            self.get_current_working_directory(), 'data', 'bin-cicids')
        if not os.path.exists(mul_cicids):
            os.makedirs(mul_cicids)
        filepath = os.path.join(
            self.get_current_working_directory(), 'data', 'bin-cicids', "train_data_features.csv")
        np.savetxt(filepath, train_data_features, delimiter=",", fmt='%.10e')
        filepath = os.path.join(
            self.get_current_working_directory(), 'data', 'bin-cicids', "train_data_results.csv")
        np.savetxt(filepath, train_data_results, delimiter=",", fmt='%.1e')
        filepath = os.path.join(
            self.get_current_working_directory(), 'data', 'bin-cicids', "test_data_features.csv")
        np.savetxt(filepath, test_data_features, delimiter=",", fmt='%.10e')
        filepath = os.path.join(
            self.get_current_working_directory(), 'data', 'bin-cicids', "test_data_results.csv")
        np.savetxt(filepath, test_data_results, delimiter=",", fmt='%.1e')

        return True

    @classmethod
    def processed_cicids_data_binary(self):
        filepath = os.path.join(
            self.get_current_working_directory(), 'data', 'bin-cicids', "train_data_features.csv")
        if not os.path.isfile(filepath):
            self.cicids_process_data_binary()


def get_current_working_directory():
    """
    get current directory
    """
    return os.getcwd()


def return_processed_cicids_data_binary():
    filepath = os.path.join(
        get_current_working_directory(), '../data', 'bin-cicids', "train_data_features.csv")

    normalized_train_data_features = np.loadtxt(filepath, delimiter=",")
    print('normalized_train_data_features finished!')
    filepath = os.path.join(
        get_current_working_directory(), '../data', 'bin-cicids', "train_data_results.csv")
    normalized_train_data_results = np.loadtxt(filepath, delimiter=",")
    print('normalized_train_data_results finished!')
    filepath = os.path.join(
        get_current_working_directory(), '../data', 'bin-cicids', "test_data_features.csv")
    normalized_test_data_features = np.loadtxt(filepath, delimiter=",")
    print('normalized_test_data_features finished!')
    filepath = os.path.join(
        get_current_working_directory(), '../data', 'bin-cicids', "test_data_results.csv")
    normalized_test_data_results = np.loadtxt(filepath, delimiter=",")
    print('normalized_test_data_results finished!')
    print(normalized_train_data_features.shape, normalized_train_data_results.shape,
          normalized_test_data_features.shape, normalized_test_data_results.shape)
    return [normalized_train_data_features, normalized_train_data_results, normalized_test_data_features,
            normalized_test_data_results]




if __name__ == '__main__':
    dataprocesss = DataProcess()
    dataprocesss.processed_cicids_data_binary()
    data = return_processed_cicids_data_binary()


