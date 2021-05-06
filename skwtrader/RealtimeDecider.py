from tqsdk import TqApi, TqAuth, tafunc, TqKq, TqAccount
import xgboost as xgb
import numpy as np
import json

import pandas as pd
import time
import sys
sys.path.append("E:/coderepo/skwquant/")
from skwutils import RawList
from trainer import DataRetriever
pd.set_option('display.max_columns', None)

html_filepath = RawList.html_filepath
api = TqApi(TqKq(), auth=TqAuth("keviniiii", "qkyhpys,1"))
# api = TqApi(TqAccount(u"G国海良时", "88500007", "702211"), auth=TqAuth("keviniiii", "qkyhpys,1"))


def get_predict_result(predict_model, pid, sequence):
    pinzhong = DataRetriever.PinzhongData(pid)
    pinzhong.update_last_feature(sequence=sequence)
    features = [pinzhong.X]
    result = predict_model.predict_proba(np.array(features))[0]
    # print(result)
    result_c = np.array(result).argmax()
    result_prob = int(result[result_c] * 100)
    result_c = int(result_c - 1)
    v_prone_percent = result[2]*100/(result[2]+result[0])
    if v_prone_percent > 70:
        result_prone = 1
    elif v_prone_percent < 30:
        result_prone = -1
    else:
        result_prone = 0
    return result_c, result_prob, result_prone


def get_display_json_and_order(symbol_raw, pinzhong_obj, vc, v_prob, v_prone):
    pid = RawList.get_name_from_symbol(symbol_raw)
    cname = RawList.pinzhong_total_list[pid]['name']
    cgroup = RawList.pinzhong_total_list[pid]['group']
    buynum = round(20000 / (list(pinzhong_obj['close'])[-1] * float(RawList.pinzhong_total_list[pid]['wan2'])), 0)
    v_prone_str = "==="
    if v_prone > 0:
        v_prone_str = "+++"
    if v_prone < 0:
        v_prone_str = "---"

    unit = {"cname": cname, "symbol": symbol_raw, "ans": vc, "prob": v_prob, "prone": v_prone_str, "v_prone": v_prone,
            "price": str(round(list(pinzhong_obj['close'])[-1], 2)), "group": str(cgroup),
            "time": str(list(pinzhong_obj['datetime'])[-1]), "buynum": str(buynum)}

    return unit


def get_all_holding_set():
    global api
    result_set = set()
    position_list = api.get_position()
    item_list = list(position_list.keys())
    for holding_item in item_list:
        holding_position = api.get_position(holding_item)
        pos_long_his = holding_position.pos_long_his
        pos_long_today = holding_position.pos_long_today
        pos_short_his = holding_position.pos_short_his
        pos_short_today = holding_position.pos_short_today
        if pos_long_his + pos_long_today + pos_short_his + pos_short_today > 0:
            result_set.add(holding_item)
    return result_set


def output_account_status_json():
    global api
    print("========================")
    account = api.get_account()
    result_obj = {"overall": {}, "data": []}
    result_obj['overall']['balance'] = str(round(account.balance, 2))
    result_obj['overall']['available'] = str(round(account.available, 2))
    result_obj['overall']['margin'] = str(round(account.margin, 2))
    result_obj['overall']['riskratio'] = str(round(account.risk_ratio*100, 2)) + "%"
    position_list = api.get_position()
    item_list = list(position_list.keys())
    print(api.get_account().balance)
    for holding_item in item_list:
        holding_position = api.get_position(holding_item)
        profit = holding_position.float_profit
        pos_long_his = holding_position.pos_long_his
        pos_long_today = holding_position.pos_long_today
        pos_short_his = holding_position.pos_short_his
        pos_short_today = holding_position.pos_short_today
        long_vol = pos_long_his + pos_long_today
        short_vol = pos_short_his + pos_short_today
        if long_vol - short_vol > 0:
            direction = "多"
        else:
            direction = "空"

        json_unit = {'symbol': holding_item,
                     'name': RawList.pinzhong_total_list[RawList.get_name_from_symbol(holding_item)]['name'],
                     'direction': direction,
                     'holding': max(long_vol, short_vol),
                     'profit': profit}
        if pos_long_his + pos_long_today + pos_short_his + pos_short_today > 0:
            result_obj['data'].append(json_unit)
            print(holding_item, profit, pos_long_his, pos_long_today, pos_short_his, pos_short_today)

    with open(RawList.quote_filepath, "w") as json_fp:
        json.dump(result_obj, json_fp)


if __name__ == '__main__':
    model = xgb.XGBClassifier()
    model.load_model('../model_repo/allset_portion_05.20210404.0.7.model')

    # klines = api.get_kline_serial(RawList.realtime_tq_symbols, 300, 1000)
    # print(klines)
    while True:
        kline_list = []
        for tmp_symbol in RawList.realtime_tq_symbols:
            tmp_kline = api.get_kline_serial(tmp_symbol, 300, 1000)
            kline_list.append(tmp_kline)
        api.wait_update()
        data_sequence = []
        result_json_obj = {'data': []}
        for klines in kline_list:
            kline_time = tafunc.time_to_datetime(klines.iloc[-1]["datetime"])
            print(kline_time)
            item_unit = pd.DataFrame()
            item_unit['datetime'] = pd.to_datetime(klines['datetime']) + pd.Timedelta('08:00:00')

            item_symbol_raw = klines.iloc[-1]["symbol"]
            item_unit['close'] = klines['close']
            vc_real, v_prob_real, v_prone_real = get_predict_result(model, item_symbol_raw, item_unit)
            # print(vc_real, v_prob_real, v_prone_real)

            tmp_json_unit = get_display_json_and_order(item_symbol_raw, item_unit, vc_real, v_prob_real, v_prone_real)
            data_sequence.append(tmp_json_unit)
        output_account_status_json()
        data_sequence.sort(key=lambda x: x['prob'], reverse=True)
        for i, item in enumerate(data_sequence):
            item['line'] = 'line' + str(i // 4 + 1)
            item['row'] = 'row' + str(i)
        result_json_obj['data'] = data_sequence
        with open(html_filepath, "w") as fp:
            json.dump(result_json_obj, fp)
        time.sleep(30)
