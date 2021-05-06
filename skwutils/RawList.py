import numpy as np
import matplotlib as plt
import random
import os
import pandas as pd
import datetime
import re

in_production = False

html_dir = "C:/nginx-1.17.10/html/"
if not in_production:
    html_dir = "../html/"
html_filepath = html_dir + "list.json"
quote_filepath = html_dir + "quote.json"

realtime_symbols = [
    'm2109.DCE', 'y2109.DCE', 'p2109.DCE', 'AP110.CZCE',
    'jd2109.DCE', 'FG109.CZCE', 'ru2109.SHFE', 'SR109.CZCE', 'CF109.CZCE',
    'TA109.CZCE', 'MA109.CZCE', 'v2109.DCE', 'eb2109.DCE', 'pp2109.DCE', 'bu2106.SHFE',
    'ZC109.CZCE', 'hc2110.SHFE', 'jm2109.DCE', 'i2109.DCE', 'rb2110.SHFE'
]
realtime_tq_symbols = [
    'DCE.m2109', 'DCE.y2109', 'DCE.p2109', 'CZCE.AP110',
    'DCE.jd2109', 'CZCE.FG109', 'SHFE.ru2109', 'CZCE.SR109', 'CZCE.CF109',
    'CZCE.TA109', 'CZCE.MA109', 'DCE.v2109', 'DCE.eb2109', 'DCE.pp2109', 'SHFE.bu2106',
    'CZCE.ZC109', 'SHFE.hc2110', 'DCE.jm2109', 'DCE.i2109', 'SHFE.rb2110'
]
'''realtime_tq_symbols = [
    'DCE.m2109'
]'''

raw_symbols_total = [
    [
        'lh000.DCE', 'c9000.DCE', 'cs000.DCE', 'a9000.DCE',
        'm9000.DCE', 'y9000.DCE', 'p9000.DCE', 'jd000.DCE', 'pg000.DCE',
        'eb000.DCE', 'l9000.DCE', 'v9000.DCE', 'pp000.DCE', 'j9000.DCE',
        'jm000.DCE', 'i9000.DCE'
    ],
    [
        'rb000.SHFE', 'cu000.SHFE', 'al000.SHFE', 'zn000.SHFE', 'pb000.SHFE',
        'sn000.SHFE', 'au000.SHFE', 'ag000.SHFE', 'hc000.SHFE', 'fu000.SHFE',
        'bu000.SHFE', 'ru000.SHFE', 'sp000.SHFE'],
    [
        'SR000.CZCE', 'CF000.CZCE', 'AP000.CZCE', 'RM000.CZCE', 'CJ000.CZCE',
        'ZC000.CZCE', 'TA000.CZCE', 'MA000.CZCE', 'FG000.CZCE', 'SF000.CZCE',
        'SM000.CZCE', 'SA000.CZCE'],

]

pinzhong_total_list = {
    "lh": {
        "name": "生猪",
        "group": "软商",
        "market": "DCE",
        "wan2": "3.21"
    },
    "c": {
        "name": "玉米",
        "group": "农产",
        "market": "DCE",
        "wan2": "1.47"
    },
    "cs": {
        "name": "淀粉",
        "group": "农产",
        "market": "DCE",
        "wan2": "1.1"
    },
    "a": {
        "name": "豆一",
        "group": "农产",
        "market": "DCE",
        "wan2": "1.7"
    },

    "m": {
        "name": "豆粕",
        "group": "农产",
        "market": "DCE",
        "wan2": "1.3"
    },
    "y": {
        "name": "豆油",
        "group": "农产",
        "market": "DCE",
        "wan2": "1.3"
    },
    "p": {
        "name": "棕榈",
        "group": "农产",
        "market": "DCE",
        "wan2": "1.5"
    },
    "jd": {
        "name": "鸡蛋",
        "group": "软商",
        "market": "DCE",
        "wan2": "1.3"
    },
    "pg": {
        "name": "液化气",
        "group": "能化",
        "market": "DCE",
        "wan2": "3.2"
    },
    "eb": {
        "name": "苯乙烯",
        "group": "能化",
        "market": "DCE",
        "wan2": "0.85"
    },
    "l": {
        "name": "塑料",
        "group": "能化",
        "market": "DCE",
        "wan2": "0.65"
    },
    "v": {
        "name": "PVC",
        "group": "能化",
        "market": "DCE",
        "wan2": "0.65"
    },
    "pp": {
        "name": "聚丙烯",
        "group": "能化",
        "market": "DCE",
        "wan2": "0.65"
    },
    "j": {
        "name": "焦炭",
        "group": "黑色",
        "market": "DCE",
        "wan2": "15"
    },
    "jm": {
        "name": "焦煤",
        "group": "黑色",
        "market": "DCE",
        "wan2": "9"
    },
    "i": {
        "name": "铁矿",
        "group": "黑色",
        "market": "DCE",
        "wan2": "17"
    },
    "rb": {
        "name": "螺纹",
        "group": "黑色",
        "market": "SHFE",
        "wan2": "1.2"
    },
    "cu": {
        "name": "铜",
        "group": "有色",
        "market": "SHFE",
        "wan2": "0.7"
    },
    "al": {
        "name": "铝",
        "group": "有色",
        "market": "SHFE",
        "wan2": "0.7"
    },
    "zn": {
        "name": "锌",
        "group": "有色",
        "market": "SHFE",
        "wan2": "0.7"
    },
    "pb": {
        "name": "铅",
        "group": "有色",
        "market": "SHFE",
        "wan2": "0.7"
    },
    "sn": {
        "name": "锡",
        "group": "有色",
        "market": "SHFE",
        "wan2": "0.14"
    },
    "au": {
        "name": "黄金",
        "group": "贵金属",
        "market": "SHFE",
        "wan2": "140"
    },
    "ag": {
        "name": "白银",
        "group": "贵金属",
        "market": "SHFE",
        "wan2": "2.7"
    },
    "hc": {
        "name": "热卷",
        "group": "黑色",
        "market": "SHFE",
        "wan2": "1.2"
    },
    "fu": {
        "name": "燃油",
        "group": "能化",
        "market": "SHFE",
        "wan2": "1.5"
    },
    "bu": {
        "name": "沥青",
        "group": "能化",
        "market": "SHFE",
        "wan2": "1.5"
    },
    "ru": {
        "name": "橡胶",
        "group": "软商",
        "market": "SHFE",
        "wan2": "1.5"
    },
    "sp": {
        "name": "纸浆",
        "group": "软商",
        "market": "SHFE",
        "wan2": "1.2"
    },
    "SR": {
        "name": "白糖",
        "group": "软商",
        "market": "CZCE",
        "wan2": "1.2"
    },
    "CF": {
        "name": "棉花",
        "group": "软商",
        "market": "CZCE",
        "wan2": "0.6"
    },
    "AP": {
        "name": "苹果",
        "group": "农产",
        "market": "CZCE",
        "wan2": "1.2"
    },
    "RM": {
        "name": "菜粕",
        "group": "农产",
        "market": "CZCE",
        "wan2": "1"
    },

    "ZC": {
        "name": "动力煤",
        "group": "黑色",
        "market": "CZCE",
        "wan2": "13"
    },
    "TA": {
        "name": "PTA",
        "group": "能化",
        "market": "CZCE",
        "wan2": "0.6"
    },
    "MA": {
        "name": "甲醇",
        "group": "能化",
        "market": "CZCE",
        "wan2": "1.3"
    },
    "FG": {
        "name": "玻璃",
        "group": "软商",
        "market": "CZCE",
        "wan2": "2.4"
    },
    "SF": {
        "name": "硅铁",
        "group": "黑色",
        "market": "CZCE",
        "wan2": "0.55"
    },
    "SM": {
        "name": "锰硅",
        "group": "黑色",
        "market": "CZCE",
        "wan2": "0.55"
    },
    "SA": {
        "name": "纯碱",
        "group": "能化",
        "market": "CZCE",
        "wan2": "2.4"
    }
}


def calc_X_y(x_raw, y_raw):
    feature_names, features = calc_X(x_raw)
    feature_names.append('vv')
    features.append(calc_y(y_raw))
    return feature_names, features


def calc_X(x_raw):
    feature_names = []
    features = []
    # feature squized array
    squiz_names, squiz_features = array_squiz(x_raw)
    feature_names.extend(squiz_names)
    features.extend(squiz_features)
    # features polyfit
    poly_names, poly_features = polyfit(x_raw)
    feature_names.extend(poly_names)
    features.extend(poly_features)
    # position
    posi_names, posi_features = posi_calc(x_raw)
    feature_names.extend(posi_names)
    features.extend(posi_features)

    return feature_names, features


def calc_y(y_raw):
    last_value = y_raw[0]
    future_value = np.array(y_raw[1:])
    f_max = max(future_value)
    f_min = min(future_value)
    result = (last_value - f_min) * 100 / (f_max - f_min)
    vv = 1
    if result > 70:
        vv = 2
    elif result < 30:
        vv = 0
    else:
        vv = 1
    return vv


def array_squiz(x_raw):
    feature_names = []
    merged_array = []
    tmp_value = 0
    for idx, value in enumerate(x_raw):
        if idx == 0:
            tmp_value = value
        else:
            if value * tmp_value < 0:
                merged_array.append(tmp_value)
                tmp_value = value
            else:
                tmp_value = tmp_value + value
    merged_array.append(tmp_value)
    sorted_array = sorted(abs(np.array(merged_array)))
    thres = sorted_array[-10]
    squiz_array = np.array(merged_array)
    squiz_array = squiz_array[abs(squiz_array) >= abs(thres)]
    features = squiz_array[-10:]
    for idx in range(0, len(features)):
        feature_names.append('s' + str(idx))
    # print(feature_names, features)
    # plot_array(merged_array)
    return feature_names, features


def posi_calc(x_raw):
    length = len(x_raw)
    current_value = x_raw[-1]
    features = []
    feature_names = []
    feature_names.append('scope_diff')
    features.append(round(max(x_raw) - min(x_raw), 2))
    for idx in range(0, 4):
        tmp_length = length // (2 ** idx)
        feature_names.append('posi' + str(tmp_length))
        raw_array = np.array(x_raw[-tmp_length:])
        max_value = max(raw_array)
        min_value = min(raw_array)

        posi = round((current_value - min_value) * 100 / (max_value - min_value), 0)
        features.append(posi)

    return feature_names, features


def plot_array(value_array, label="line1"):
    length = len(list(value_array))
    x = range(0, length)
    y = list(value_array)
    plt.plot(x, y, label=label)
    plt.legend()
    plt.show()


def polyfit(x_raw):
    raw_array = np.add.accumulate(x_raw) + 10000
    raw_array = raw_array * 100 / float(raw_array[0])

    resultmax = np.max(np.where(raw_array == np.max(raw_array)))
    resultmin = np.max(np.where(raw_array == np.min(raw_array)))
    start_idx = min(resultmin, resultmax)
    length = len(raw_array) - start_idx
    x = list(range(0, length))
    y = raw_array[-length:]
    weights = np.polyfit(x, y, 2)
    result = 2 * weights[0] * x[-1] + weights[1]
    result = result * 100
    # print(result)
    if random.random() < 1 and False:
        print(len(raw_array))
        model = np.poly1d(weights)
        x_output = x
        y1 = list(y)
        y2 = model(x_output)
        plt.plot(x_output, y1, label="line 1")
        plt.plot(x_output, y2, label="line 2")
        plt.legend()
        plt.show()
    return ["polyslope"], [round(result * 100, 0)]


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


def check_time(time_to_check):
    on_time = datetime.time(8, 55)
    off_time = datetime.time(11, 35)
    if on_time <= time_to_check <= off_time:
        return True
    on_time = datetime.time(13, 25)
    off_time = datetime.time(15, 5)
    if on_time <= time_to_check <= off_time:
        return True
    on_time = datetime.time(20, 55)
    off_time = datetime.time(23, 55)
    if on_time <= time_to_check <= off_time:
        return True
    return False


def get_name_from_symbol(symbol):
    sub_symbol = symbol.split(".")[1]
    m = re.search(r"\d", sub_symbol)
    start = int(m.start())
    return sub_symbol[0:start]
