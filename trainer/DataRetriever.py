import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from skwutils import RawList
import os
pd.options.display.max_rows = 200
pd.set_option('display.min_rows', 200)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)


class PinzhongData:
    def __init__(self, pid):
        self.pid = pid
        self.sequence = pd.DataFrame()
        self.evidence_len = 1000
        self.future_len = 100
        self.feature_names = []
        self.training_list = []
        self.X = []
        self.y = []

    def recalculate(self, training_data_dir='history/', feature_dir='features/'):
        self.load_close_sequence(training_data_dir)
        self.sequence['pct'] = round(self.sequence['close'].pct_change() * 10000, 0)
        self.sequence = self.sequence.dropna()
        self.format_to_evidence_label_raw_array()
        self.sequence.set_index('datetime')
        self.calculate_training_list()
        # print(self.sequence)
        self.training_list.to_csv(feature_dir + self.pid + '.csv', index=False)
        self.get_X_y()

    def load_features_from_file(self, feature_dir):
        self.training_list = pd.read_csv(feature_dir + self.pid + '.csv')
        self.feature_names = self.training_list.columns.tolist()
        print(self.feature_names)
        self.get_X_y()

    def update_last_feature(self, realtime_dir=None, sequence=None):
        if sequence is not None:
            self.sequence = sequence
        if realtime_dir is not None:
            self.load_close_sequence(realtime_dir)
        self.sequence['pct'] = round(self.sequence['close'].pct_change() * 10000, 0)
        self.sequence = self.sequence.dropna()
        self.feature_names, self.X = self.get_last_feature()

    def load_close_sequence(self, filepath):
        filename = filepath + self.pid + '.csv'
        self.sequence = pd.read_csv(filename)
        self.sequence.set_index('datetime')

    def format_to_evidence_label_raw_array(self):
        close_array = self.sequence['pct'].tolist()
        array_length = len(close_array)
        total_length = self.evidence_len + self.future_len
        for idx in range(total_length, array_length):
            raw_list = close_array[idx-total_length:idx]
            evid_list = raw_list[0: self.evidence_len]
            label_list = raw_list[self.evidence_len-1:]
            self.training_list.append([evid_list, label_list, []])

    def get_X_y(self):
        label_sequence = self.training_list[['vv']]
        evidence_sequence = self.training_list.copy()
        cols = ['vv']
        evidence_sequence = evidence_sequence.drop(cols, axis=1)
        self.X = np.array(pd.DataFrame(evidence_sequence))
        self.y = np.array(label_sequence['vv'])

    def calculate_training_list(self):
        for item in self.training_list:
            X_raw = item[0]
            y_raw = item[1]
            self.feature_names, features = RawList.calc_X_y(X_raw, y_raw)
            item[2] = features

        self.training_list = pd.DataFrame(self.training_list, columns=['X_raw', 'y_raw', 'features'])
        self.training_list = pd.DataFrame(self.training_list['features'].tolist(), columns=self.feature_names)
        print(self.training_list)

    def get_last_feature(self):
        close_array = self.sequence['pct'].tolist()
        evid_list = close_array[-1000:]
        feature_names, features = RawList.calc_X(evid_list)
        return feature_names, features

    def down_sampling_features(self, portion=0.1):
        X_raw = self.X
        y_raw = self.y
        result_X = []
        result_y = []
        for idx in range(0, len(y_raw)):
            if random.random() < portion:
                result_X.append(X_raw[idx])
                result_y.append(y_raw[idx])

        result_X = np.array(result_X)
        result_y = np.array(result_y)
        return result_X, result_y








if __name__ == '__main__':
    '''
    Task List
    0 - Generate Training Data
    1 - Load a file
    2 - convert txt to csv
    
    '''
    task_id = 2
    if task_id == 1:
        for pinzhong_list in RawList.raw_symbols_total:
            for pinzhong in pinzhong_list:
                data = PinzhongData(pinzhong, load_type='recalculate', data_dir='history/', feature_dir='features/')
    if task_id == 2:
        RawList.convert_text_csv()