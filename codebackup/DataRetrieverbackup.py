import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import tbpy
import time
from skwutils import RawList
import datetime
pd.options.display.max_rows = 200
pd.set_option('display.min_rows', 200)
tbpy.init()

def polyfit(value_array):
    resultmax = np.max(np.where(value_array == np.max(value_array)))
    resultmin = np.max(np.where(value_array == np.min(value_array)))
    start_idx = min(resultmin, resultmax)
    length = len(value_array) - start_idx
    x = list(range(0, length))
    y = value_array[-length:]
    weights, residuals, rank, singular_values, rcond= np.polyfit(x, y, 2, full=True)
    result = (2*weights[0]*x[-1] + weights[1])*1000/list(y)[-1]
    # print(result)
    if random.random() < 0.01 and False:
        model = np.poly1d(weights)
        x_output = x
        y1 = list(y)
        y2 = model(x_output)
        plt.plot(x_output, y1, label="line 1")
        plt.plot(x_output, y2, label="line 2")
        plt.legend()
        plt.show()
    return result


def change_pct(valuearray):
    return round((valuearray[-1]-valuearray[0])*100/valuearray[0],2)

class PinzhongData:
    def __init__(self, pid, load_type='recalculate', data_dir='history/', feature_dir='features/', realtime_dir='latest/'):

        self.pid = pid
        # self.sequence = self.load_data_sina()
        if load_type == 'recalculate':
            self.sequence = pd.DataFrame()
            self.load_data_tb(data_dir)
            self.reformat_tb(5)
            self.extend_sequence(200, width=2.5)
            print(self.sequence.tail(1).to_dict())
            self.sequence = self.sequence.dropna()
            self.sequence.to_csv(feature_dir + pid + '.csv', index=False)
            self.get_evidence_label_sequence()
        elif load_type == 'load_file':
            self.sequence = pd.read_csv(feature_dir + pid + '.csv')
            self.get_evidence_label_sequence()
        elif load_type == 'realtime':
            self.sequence = pd.DataFrame()
            self.load_data_tb(realtime_dir)
            self.reformat_tb(5)
            self.X = self.get_last_feature()


        # self.sequence = self.sequence.drop(['pct'], axis=1)

    def load_data_tb(self, filepath):
        filename = filepath + self.pid + '.csv'
        raw_data = pd.read_csv(filename)
        self.sequence['datetime'] =pd.to_datetime(raw_data['time'])
        self.sequence['open'] = raw_data['open']
        self.sequence['close'] = raw_data['close']
        self.sequence.set_index('datetime')

    def reformat_tb(self, wan_pct):
        df = self.sequence
        new_frame_array = []
        for idx, row in df.iterrows():

            local_open = row['open']
            local_close = row['close']
            price_step_percent = (local_close - local_open)/ local_open * 10000
            step = int(max(abs(round(price_step_percent / wan_pct)), 1))
            price_step = (local_close - local_open) / step
            time_delta = pd.to_timedelta(str(300 // step) + ' seconds')
            start_time = row['datetime']
            for sub_time_posi in range(0, step):
                internal_close = round(local_open + sub_time_posi*price_step, 2)
                local_time = start_time + sub_time_posi*time_delta
                new_frame_array.append([local_time, internal_close])
        self.sequence = pd.DataFrame(new_frame_array, columns=['datetime', 'close'])
        self.sequence['pct'] = self.sequence['close'].pct_change() * 10000
        self.sequence.set_index('datetime')

    def get_evidence_label_sequence(self):
        self.evidence_sequence = self.sequence.copy()
        cols = ['datetime', 'close', 'lastmax', 'lastmin', 'vv']
        self.label_sequence = self.sequence[['vv']]
        self.evidence_sequence = self.evidence_sequence.drop(cols, axis=1)
        self.evidence_label = self.evidence_sequence.columns
        if 'Unnamed' in self.evidence_label[0]:
            self.evidence_label = self.evidence_label[1:]
        self.X = np.array(pd.DataFrame(self.evidence_sequence))
        self.y_class = np.array(self.label_sequence['vc'])
        self.y_value = np.array(self.label_sequence['vv'])

    def extend_sequence(self, future_num):
        self.sequence['lastmax'] = self.sequence['close'].rolling(future_num).max().shift(-future_num)
        self.sequence['lastmin'] = self.sequence['close'].rolling(future_num).min().shift(-future_num)
        self.sequence['avg'] = self.sequence['close'].rolling(future_num).mean().shift(-future_num)
        self.sequence['vv'] = round((self.sequence['lastmax'] if self.sequence['avg']>=0 else self.sequence['lastmin'])*100/self.sequence['close'], 2)
        print(self.sequence)
        self.sequence['polyfit'] = self.sequence['close'].rolling(2000).apply(polyfit)
        for idx in range(4, 14):
            period = 2**idx
            self.sequence['x'+str(period)] = self.sequence['close'].rolling(period).apply(change_pct)


    def get_last_feature(self):
        pf_array = self.sequence['close'][-2000:]
        features = []
        pf = polyfit(pf_array)
        features.append(pf)
        for idx in range(4, 14):
            period = 2 ** idx
            change = self.sequence['pct'][-period:].sum()
            features.append(change)
        return np.array(features)

    def get_features(self, portion=0.1):
        X_raw = self.X
        y_raw = self.y_class + 1
        result_X = []
        result_y = []
        for idx in range(0, len(y_raw)):
            if random.random() < portion:  # and y_raw[idx] != 0:
                result_X.append(X_raw[idx])
                result_y.append(y_raw[idx])
        result_X = np.array(result_X)
        result_y = np.array(result_y)
        return result_X, result_y


def retrieve(symbol_list, data_dir='history/', frequency='5m'):
    # bars = tbpy.get_history_n(symbol_list, frequency, 10000, fields=None, timeout='20s')
    bars = tbpy.get_history(symbol=symbol_list, frequency=frequency, begin_time=datetime(2021, 1, 10),
                     end_time=datetime(2021, 4, 19))
    for item in symbol_list:
        data = pd.DataFrame(bars[item])
        data.to_csv(data_dir + item + ".csv")


def prepare_training_data(data_dir, frequency):
    for item_list in RawList.raw_symbols_total:
        retrieve(item_list, data_dir=data_dir, frequency=frequency)
        time.sleep(70)


if __name__ == '__main__':
    '''
    Task List
    0 - Generate Training Data
    1 - 
    
    '''
    task_id = 0
    if task_id == 0:
        prepare_training_data("history/", "5m")
