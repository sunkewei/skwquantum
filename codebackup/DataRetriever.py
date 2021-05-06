import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import tbpy
import time
from skwutils import RawList
import os

pd.options.display.max_rows = 200
pd.set_option('display.min_rows', 200)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)


def polyfit(value_array):
    resultmax = np.max(np.where(value_array == np.max(value_array)))
    resultmin = np.max(np.where(value_array == np.min(value_array)))
    start_idx = min(resultmin, resultmax)
    length = len(value_array) - start_idx
    x = list(range(0, length))
    y = value_array[-length:]
    weights, residuals, rank, singular_values, rcond = np.polyfit(x, y, 2, full=True)
    result = (2 * weights[0] * x[-1] + weights[1]) * 1000 / list(y)[-1]
    # print(result)
    if random.random() < 0.1 and False:
        print(len(value_array))
        model = np.poly1d(weights)
        x_output = x
        y1 = list(y)
        y2 = model(x_output)
        plt.plot(x_output, y1, label="line 1")
        plt.plot(x_output, y2, label="line 2")
        plt.legend()
        plt.show()
    return result


def polyfit_diff(value_array):
    resultmax = np.max(np.where(value_array == np.max(value_array)))
    resultmin = np.max(np.where(value_array == np.min(value_array)))
    start_idx = min(resultmin, resultmax)
    length = len(value_array) - start_idx
    x = list(range(0, length))
    y = value_array[-length:]
    weights, residuals, rank, singular_values, rcond = np.polyfit(x, y, 2, full=True)
    model = np.poly1d(weights)
    x_output = x[-1]
    y = y.tolist()
    result = round((model(x_output) - y[-1]) / y[-1], 4)
    return result


def getValueClass(value_array):
    tmp_array = value_array.tolist()
    width = 1.0
    if tmp_array[0] > width:
        return 1
    elif tmp_array[0] < -width:
        return -1
    else:
        return 0


def tunnel_width(maxb, mina, width):
    result_class = -2
    result_value = 0
    if mina * maxb >= 0:
        # 单边上升
        if mina >= 0:
            result_class = 1
            result_value = maxb
        # 单边下降
        if maxb <= 0:
            result_class = -1
            result_value = mina
    else:

        if abs(maxb) / abs(mina) > 2 and abs(maxb) > width * 0.7:
            result_class = 1
            result_value = maxb
        elif abs(maxb) / abs(mina) < 0.5 and abs(mina) > width * 0.7:
            result_class = -1
            result_value = mina
        else:
            result_class = 0
            result_value = min(abs(maxb), abs(mina))
    return result_class, result_value


def pct_change(a, b):
    return round((b * 100 / a - 100), 2)


class PinzhongData:
    def __init__(self, pid, load_type='recalculate', data_dir='history/', feature_dir='features/',
                 realtime_dir='latest/'):

        self.pid = pid
        # self.sequence = self.load_data_sina()
        if load_type == 'recalculate':
            self.sequence = pd.DataFrame()
            self.load_data_tb(data_dir)
            self.reformat_tb(5)
            self.extend_sequence(120)
            # print(self.sequence)
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
        # raw_data = pd.read_csv(filename, names=['time', 'open', 'high', 'low', 'close', 'vol', 'holding', 'jiesuan'])
        raw_data = pd.read_csv(filename)
        self.sequence['datetime'] = pd.to_datetime(raw_data['time'])
        self.sequence['open'] = raw_data['open']
        self.sequence['close'] = raw_data['close']
        self.sequence.set_index('datetime')
        # fig = px.line(self.sequence, x="datetime", y="close", title=self.pid)
        # fig.show()
        pass

    def reformat_tb(self, wan_pct):
        df = self.sequence
        new_frame_array = []
        for idx, row in df.iterrows():

            local_open = row['open']
            local_close = row['close']
            price_step_percent = (local_close - local_open) / local_open * 10000
            step = int(max(abs(round(price_step_percent / wan_pct)), 1))
            price_step = (local_close - local_open) / step
            time_delta = pd.to_timedelta(str(300 // step) + ' seconds')
            start_time = row['datetime']
            for sub_time_posi in range(0, step):
                internal_close = round(local_open + sub_time_posi * price_step, 2)
                local_time = start_time + sub_time_posi * time_delta
                new_frame_array.append([local_time, internal_close])
        self.sequence = pd.DataFrame(new_frame_array, columns=['datetime', 'close'])
        self.sequence['pct'] = self.sequence['close'].pct_change() * 10000
        self.sequence.set_index('datetime')

    def get_evidence_label_sequence(self):
        self.evidence_sequence = self.sequence.copy()
        cols = ['datetime', 'close', 'pct', 'vv', 'vc', 'avg']
        self.label_sequence = self.sequence[['vv', 'vc']]
        self.evidence_sequence = self.evidence_sequence.drop(cols, axis=1)
        self.evidence_label = self.evidence_sequence.columns
        if 'Unnamed' in self.evidence_label[0]:
            self.evidence_label = self.evidence_label[1:]
        self.X = np.array(pd.DataFrame(self.evidence_sequence))
        self.y_value = np.array(self.label_sequence['vv'])
        self.y_class = np.array(self.label_sequence['vc'])

    def extend_sequence(self, future_num):
        avg = self.sequence['avg'] = pct_change(self.sequence['close'],
                                                self.sequence['close'].rolling(future_num).mean().shift(-future_num))
        self.sequence['lastmax'] = pct_change(self.sequence['close'],
                                              self.sequence['close'].rolling(future_num).max().shift(-future_num))
        self.sequence['lastmin'] = pct_change(self.sequence['close'],
                                              self.sequence['close'].rolling(future_num).min().shift(-future_num))
        self.sequence['lastmax1'] = self.sequence['lastmax'] * np.sign(self.sequence['avg'])
        self.sequence['lastmin1'] = self.sequence['lastmin'] * np.sign(self.sequence['avg'])
        self.sequence['vv'] = self.sequence[['lastmax1', 'lastmin1']].max(axis=1) * np.sign(self.sequence['avg'])
        self.sequence['vc'] = self.sequence['vv'].rolling(1).apply(getValueClass)
        cols = ['lastmax', 'lastmin', 'lastmax1', 'lastmin1']
        self.sequence = self.sequence.drop(cols, axis=1)
        polyfit_number = int(128)
        for idx in range(0, 4):
            period = polyfit_number // (2 ** idx)
            self.sequence['polyfit' + str(period)] = self.sequence['close'].rolling(period).apply(polyfit)
            if idx < 5:
                self.sequence['polydiff' + str(period)] = self.sequence['close'].rolling(period).apply(polyfit_diff)

        for idx in range(3, 8):
            period = int(2 ** idx)
            self.sequence['x' + str(period)] = self.sequence['pct'].rolling(period).sum()

    def get_last_feature(self):
        polyfit_number = 512
        pf_array = self.sequence['close'][-polyfit_number:]
        features = []
        for idx in range(0, 3):
            period = polyfit_number // (2 ** idx)
            tmp_pfarray = pf_array[-period:]
            pf = polyfit(tmp_pfarray)
            features.append(pf)
            if idx < 2:
                pf_diff = polyfit_diff(tmp_pfarray)
                features.append(pf_diff)
        for idx in range(6, 9):
            period = 2 ** idx
            change = self.sequence['pct'][-period:].sum()
            features.append(change)
        return np.array(features)

    def get_features(self, isClass, portion=0.1):

        X_raw = self.X
        y_raw = self.y_value
        if isClass:
            y_raw = self.y_class

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
    bars = tbpy.get_history_n(symbol_list, frequency, 10000, fields=None, timeout='20s')
    # bars = tbpy.get_history(symbol=symbol_list, frequency=frequency, begin_time=datetime(2021, 1, 10),
    #                         end_time=datetime(2021, 4, 19), timeout='30s')
    for item in symbol_list:
        data = pd.DataFrame(bars[item])
        data.to_csv(data_dir + item + ".csv")


def prepare_training_data(data_dir, frequency):
    for item_list in RawList.raw_symbols_total:
        retrieve(item_list, data_dir=data_dir, frequency=frequency)
        time.sleep(70)


def convert_text_csv():
    file_list = os.listdir("history/")
    for filename in file_list:
        if filename.endswith('.txt'):
            with open('history/' + filename, "r") as fp:
                lines = fp.readlines()
                lines = lines[2:-1]
                total_lines = []
                for line in lines:
                    line_raw = line.split(',')
                    close = line_raw[5]
                    date_str = line_raw[0].replace("/", "-")
                    time_str = line_raw[1][:2] + ":" + line_raw[1][2:]
                    line_new = date_str + " " + time_str + "," + str(close)
                    total_lines.append(line_new + "\n")

            with open('history/' + filename[:-4] + ".csv", "w") as fp:
                fp.writelines(total_lines)
            raw_data = pd.read_csv('history/' + filename[:-4] + ".csv",
                                   names=['datetime', 'close'])
            raw_data.to_csv('history/' + filename[:-4] + ".csv")


if __name__ == '__main__':
    '''
    Task List
    0 - Generate Training Data
    1 - Load a file
    2 - convert txt to csv

    '''
    task_id = 2
    if task_id == 0:
        tbpy.init()
        prepare_training_data("history/", "5m")
    if task_id == 1:
        for pinzhong_list in RawList.raw_symbols_total:
            for pinzhong in pinzhong_list:
                data = PinzhongData(pinzhong, load_type='recalculate', data_dir='history/', feature_dir='features/')
    if task_id == 2:
        convert_text_csv()