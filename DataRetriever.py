from urllib import request
import json
import numpy as np
import pandas as pd
import plotly.express as px
from scipy.signal import find_peaks
import plotly.graph_objects as go
import tbpy


pd.options.display.max_rows = 200
pd.set_option('display.min_rows', 200)


def polyfit(value_array):
    resultmax = np.max(np.where(value_array == np.max(value_array)))
    resultmin = np.max(np.where(value_array == np.min(value_array)))
    start_idx = min(resultmin, resultmax)
    length = len(value_array) - start_idx
    x = list(range(0,length))
    y = value_array[-length:]
    weights = np.polyfit(x, y, 2)
    model = np.poly1d(weights)
    result = model(length+5)

    # length = ((resultmax-resultmin)/abs(resultmax-resultmin))*length
    return result


class PinzhongData:
    def __init__(self, pid):
        self.pid = pid
        self.sequence = self.load_data()
        self.update_max(5)

    def get_data_from_sina(self, time_span):
        url = 'http://stock2.finance.sina.com.cn/futures/api/json.php/IndexService.getInnerFuturesMiniKLine' + str(
            time_span) + 'm?symbol='
        url = url + self.pid
        print(url)
        req = request.Request(url)
        rsp = request.urlopen(req)
        res = rsp.read()
        res_json = json.loads(res)
        res_json.reverse()
        res_json_refined = []

        for idx, item in enumerate(list(res_json)):
            if idx==0:
                continue
            if int(item[5])<int(res_json[idx-1][5])/20:
                if item[2]==item[3] and item[1]==item[2] and item[3]==item[4]:
                    # print("removed ------------------------------------", item)
                    continue
            else:
                res_json_refined.append(item)


        if time_span < 60:
            for item in res_json_refined:
                item.append(1)
        else:
            for item in res_json_refined:
                [hour, minutes, _] = item[0].split(" ")[1].split(":")
                if (hour == '14' and minutes == '00') or minutes == '30':
                    item.append(0.5)
                else:
                    item.append(1)

        res_json_refined.insert(0, ['date', 'open', 'high', 'low', 'close', 'vol','span'])
        raw_data = np.array(res_json_refined)
        df = pd.DataFrame(data=raw_data[1:], columns=raw_data[0])
        df['datetime'] = pd.to_datetime(df['date'])
        df['open'] = pd.to_numeric(df['open'])
        df['high'] = pd.to_numeric(df['high'])
        df['low'] = pd.to_numeric(df['low'])
        df['close'] = pd.to_numeric(df['close'])
        df['vol'] = pd.to_numeric(df['vol'])
        df['span'] = pd.to_numeric(df['span'])


        df = df.set_index('datetime')
        df.drop(['date'], axis=1, inplace=True)
        df.head()
        # print(df)
        return df


    @staticmethod
    def get_time_passed(current_time):
        pass

    @staticmethod
    def reformat_to_5min(df, time_span):
        new_frame_array = []

        #print(df)

        time_delta = pd.to_timedelta('5 minutes')
        for start_time, row in df.iterrows():
            local_rate = row['span']
            local_open = row['open']
            local_close = row['close']
            step = int((time_span//5)*local_rate)
            price_step = (local_close - local_open) / step
            price_step_percent = int(price_step/local_open*10000)
            step_increase_rate = int(max(abs(round(price_step_percent/3)),1))
            step = step_increase_rate*step
            price_step = (local_close - local_open) / step
            time_delta = pd.to_timedelta(str(300//step_increase_rate)+' seconds')

            for sub_time_posi in range(0, step):
                internal_close = round(local_open + sub_time_posi*price_step, 2)
                local_time = start_time + sub_time_posi*time_delta - pd.to_timedelta(str(int(time_span*local_rate)) + ' minutes')
                new_frame_array.append([local_time, internal_close])
        df = pd.DataFrame(new_frame_array, columns=['datetime', 'close'])
        df['pct'] = df['close'].pct_change()*10000
        df.set_index('datetime')
        df.head()
        return df

    def load_data_one_layer(self, time_span):
        df = self.get_data_from_sina(time_span)
        df = PinzhongData.reformat_to_5min(df, time_span)
        return df

    def load_data(self):
        df5 = self.load_data_one_layer(5)
        df15 = self.load_data_one_layer(15)
        df30 = self.load_data_one_layer(30)
        df60 = self.load_data_one_layer(60)
        start_datetime_5 = df5.loc[0]['datetime']
        start_datetime_15 = df15.loc[0]['datetime']
        start_datetime_30 = df30.loc[0]['datetime']
        mask = (df15['datetime'] < start_datetime_5)
        df15 = df15.loc[mask]
        mask = (df30['datetime'] < start_datetime_15)
        df30 = df30.loc[mask]
        mask = (df60['datetime'] < start_datetime_30)
        df60 = df60.loc[mask]
        df_total = pd.concat([df60, df30, df15, df5], ignore_index=True)
        df_total.set_index('datetime')
        df_total.sort_index()
        return df_total

    def load_from_tbquant(self):
        pass

    def update_max(self, last_num):
        # self.sequence['lastmax'] = self.sequence['close'].rolling(last_num).max()
        # self.sequence['lastmin'] = self.sequence['close'].rolling(last_num).min()
        # self.sequence['nextclose'] = self.sequence['close'].shift(-1)
        self.sequence['start'] = self.sequence['close'].rolling(2000).apply(polyfit)
        self.sequence['average'] = round(self.sequence['close'].rolling(200).mean(), 2)
        self.sequence['diff'] = round((self.sequence['close'] - self.sequence['average']), 2)
        self.sequence.drop(columns=['average'])

if __name__ == '__main__':
    load = True
    if load:
        data = PinzhongData('i2105')
        print(data.sequence)
        data.sequence.to_csv('raw.txt')
    data = pd.read_csv('raw.txt')
    fig = go.Figure()
    index_list = list(range(0,len(data['close'])))
    fig.add_trace(go.Scatter(x=index_list, y=list(data['close']), mode='lines'))
    fig.add_trace(go.Scatter(x=index_list, y=list(data['average']), mode='lines'))
    fig.add_trace(go.Scatter(x=index_list, y=list(data['start']), mode='lines', name='polyfit'))

    # fig_raw = px.line(data.sequence, y="close")
    # fig = px.histogram(data.sequence, x="start")
    fig.show()
    # fig_raw.show()


    '''
    a = np.array([1,2,3,4,5,6,5,4,3,2,1])
    print(polyfit(a, 1))
    '''

'''
reference

    idx_max, prop = find_peaks(data['diff'], height=0,  distance=200)
    idx_min, prop = find_peaks(-data['diff'], height=0, distance=200)
    print(prop)
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=list(range(0,len(data['close']))), y=list(data['close']), mode='lines'))
    fig.add_trace(go.Scatter(x=idx_max, y=list(data['close'][idx_max]), mode='markers'))
    fig.add_trace(go.Scatter(x=idx_min, y=list(data['close'][idx_min]), mode='markers'))


'''