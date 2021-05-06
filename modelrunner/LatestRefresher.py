import tbpy
import time
import datetime
import threading
import pandas as pd
from skwutils import RawList

tbpy.init()




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


def retrieve(symbol_list, frequency='5m'):
    current_time = datetime.datetime.now().time()
    doit = check_time(current_time)
    if not doit:
        print("Not in time")
        return
    bars = tbpy.get_history_n(symbol_list, frequency, 5000, fields=['time','close'], timeout='20s')
    for item in symbol_list:
        data = pd.DataFrame(bars[item])
        data = data.rename(columns={'time': 'datetime'})
        data.to_csv("latest/" + item + ".csv")


if __name__ == '__main__':
    idx = 0
    while True:
        check_thread = threading.Thread(target=retrieve, args=(RawList.realtime_symbols,))
        check_thread.start()
        print("last update", datetime.datetime.now().date(), datetime.datetime.now().time())
        time.sleep(70)






'''

begintime=datetime.datetime.strptime('20210101','%Y%m%d')
endtime=datetime.datetime.strptime('20210329','%Y%m%d')
#bars=tbpy.get_history(symbols, freq, begintime, endtime, fields=None, timeout='30s')
bars=tbpy.get_history_n(raw_symbols[1], freq, 5000, fields=None, timeout='300s')
print(bars)
for item in raw_symbols:
    data = pd.DataFrame(bars[item])
    data.to_csv("history/"+item+".csv")
    
print(data)
'''

