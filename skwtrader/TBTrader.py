import tbpy
import threading
import time
import json
from skwutils import RawList
import re
def get_name_from_symbol(symbol):
    m = re.search(r"\d", symbol)
    start = int(m.start())
    return symbol[0:start]

def update_account_status():

    position_list = account.get_position()
    json_obj = {"data":[]}
    for item in position_list:
        long_vol = item.l_current_volume
        short_vol = item.s_current_volume
        if long_vol - short_vol > 0:
            direction = "多"
        else:
            direction = "空"
        json_unit = {}
        json_unit['symbol'] = item.symbol
        json_unit['name'] = RawList.pinzhong_total_list[get_name_from_symbol(item.symbol)]['name']
        json_unit['direction'] = direction
        json_unit['holding'] = max(long_vol, short_vol)
        json_unit['profit'] = item.l_float_porfit + item.s_float_porfit
        json_obj['data'].append(json_unit)
        print(json_unit)

    with open(RawList.quote_filepath, "w") as fp:
        json.dump(json_obj, fp)



if __name__ == '__main__':
    tbpy.init()
    account = tbpy.get_account('sim004904')
    while True:
        update_thread = threading.Thread(target=update_account_status)
        update_thread.start()
        time.sleep(10)