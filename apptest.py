import tbpy
import datetime
import numpy
import pandas as pd
tbpy.init()

symbols=[
    'lh2109.DCE',
    'c2105.DCE',
    'cs2105.DCE',
    'a2105.DCE',
    'b2105.DCE',
    'm2105.DCE',
    'y2105.DCE',
    'p2105.DCE',
    'jd2105.DCE',
    'pg2105.DCE',
    'eb2105.DCE',
    'l2105.DCE',
    'v2105.DCE',
    'pp2105.DCE',
    'j2105.DCE',
    'jm2105.DCE',
    'i2105.DCE',
    'rb2105.SHFE',
    'cu2105.SHFE',
    'al2105.SHFE',
]
symbols1=[
    'zn2105.SHFE',
    'pb2105.SHFE',
    'sn2105.SHFE',
    'au2106.SHFE',
    'ag2106.SHFE',
    'hc2105.SHFE',
    'fu2105.SHFE',
    'bu2106.SHFE',
    'ru2105.SHFE',
    'sp2105.SHFE',
]
symbols2=[
    'SR105.CZCE',
    'CF105.CZCE',
    'AP105.CZCE',
    'RM105.CZCE',
    'CJ105.CZCE',
    'ZC105.CZCE',
    'TA105.CZCE',
    'MA105.CZCE',
    'FG105.CZCE',
    'SF105.CZCE',
    'SM105.CZCE',
    'SA105.CZCE',
]
freq='5m'
begintime=datetime.datetime.strptime('20210101','%Y%m%d')
endtime=datetime.datetime.strptime('20210329','%Y%m%d')
#bars=tbpy.get_history(symbols, freq, begintime, endtime, fields=None, timeout='30s')
bars=tbpy.get_history_n(symbols2, freq, 5000, fields=None, timeout='300s')
print(bars)
for item in symbols2:
    data = pd.DataFrame(bars[item])
    data.to_csv("history/"+item+".csv")
print(data)


