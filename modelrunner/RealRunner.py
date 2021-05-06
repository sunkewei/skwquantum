from trainer.DataRetriever import PinzhongData
import xgboost as xgb
import numpy as np
import json
import re
import datetime
import time
from skwutils import RawList

# html_filepath = "html/list.json"
html_filepath = RawList.html_filepath


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
    m = re.search(r"\d", symbol)
    start = int(m.start())
    return symbol[0:start]


if __name__ == '__main__':
    model = xgb.XGBClassifier()
    model.load_model('model_repo/allset_portion_05.20210404.0.7.model')

    while True:
        try:
            current_time = datetime.datetime.now().time()
            doit = True # check_time(current_time)
            if not doit:
                print("Not in time")
                time.sleep(20)
                continue
            data_sequence = []
            json_obj = {'data': []}
            for symbol in RawList.realtime_symbols:
                print("=======================", symbol)
                whole_features_list = []
                pinzhong = PinzhongData(symbol, load_type='realtime', realtime_dir='latest/')
                features = pinzhong.X
                whole_features_list.append(features)
                ans = model.predict_proba(np.array(whole_features_list))[0]
                vc = np.array(ans).argmax()
                v_prob = int(ans[vc]*100)
                vc = int(vc - 1)
                pid = get_name_from_symbol(symbol)
                cname = RawList.pinzhong_total_list[pid]['name']
                cgroup = RawList.pinzhong_total_list[pid]['group']
                buynum = round(20000 / (list(pinzhong.sequence['close'])[-1] * float(
                    RawList.pinzhong_total_list[pid]['wan2'])), 0)

                item = {"cname": cname, "symbol": str(symbol.split('.')[0]), "ans": vc, "prob": v_prob,
                        "price": str(round(list(pinzhong.sequence['close'])[-1], 2)), "group": str(cgroup),
                        "time": str(list(pinzhong.sequence['datetime'])[-1]), "buynum": str(buynum)}
                data_sequence.append(item)
                print(symbol, ans)
                print(list(pinzhong.sequence['datetime'])[-1])
            # print(data_sequence)
            data_sequence.sort(key=lambda x: x['prob'], reverse=True)
            for idx, item in enumerate(data_sequence):
                item['line'] = 'line'+str(idx//4 + 1)
                item['row'] = 'row'+str(idx)
            json_obj['data'] = data_sequence
            with open(html_filepath, "w") as fp:
                json.dump(json_obj, fp)
            time.sleep(60)
        except Exception as error:
            print("The following error occured - ", error)
            time.sleep(60)
            continue
