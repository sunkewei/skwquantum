from tqsdk import TqApi, TqAuth, TqKq, TqAccount
import json

import pandas as pd
import time
import sys
sys.path.append("E:/coderepo/skwquant/")
from skwutils import RawList
pd.set_option('display.max_columns', None)
api = TqApi(TqKq(), auth=TqAuth("keviniiii", "qkyhpys,1"))
# api = TqApi(TqAccount("G国海良时", "88500007", "702211"), auth=TqAuth("keviniiii", "qkyhpys,1"))


def open_order_json(unit, holding_set, alive_open_set):
    symbol_raw = unit['symbol']
    vc = unit['ans']
    v_prob = unit['prob']
    v_prone = unit['v_prone']
    buynum = int(float(unit['buynum']))
    price_limit = 0
    open_order(symbol_raw, vc, v_prob, v_prone, buynum,
               price_limit=price_limit, holding_set=holding_set, alive_open_set=alive_open_set)
    return unit


def open_order(symbol_raw, vc, v_prob, v_prone, volume, price_limit, holding_set, alive_open_set):
    global api
    possibility_limit = 77
    indicator = vc * v_prob
    direction = None
    if indicator > possibility_limit:
        direction = "BUY"  # open buy
    elif indicator < -possibility_limit:
        direction = "SELL"  # open sell
    elif indicator == 0 and v_prob > possibility_limit:
        if v_prone > 0:
            direction = "BUY"  # open buy
        if v_prone < 0:
            direction = "SELL"  # open sell
    if direction is not None:
        if symbol_raw in alive_open_set:
            print(symbol_raw, "open order already placed, skip")
            return

        if symbol_raw in holding_set:
            print(symbol_raw, "already on list, skip trading")
            return
        limit = price_limit
        if price_limit == 0:
            quote = api.get_quote(symbol_raw)
            api.wait_update()
            limit = quote.last_price
        api.insert_order(symbol=symbol_raw, direction=direction, offset="OPEN", limit_price=limit, volume=volume)
        api.wait_update()
        print(symbol_raw, "开仓指令发出", direction)


def close_order(symbol_raw, profit, pos_long_his, pos_long_today, pos_short_his, pos_short_today, alive_close_set):
    global api

    if symbol_raw in alive_close_set:
        print(symbol_raw, "close order already placed, skip")
        return
    if profit > 400:
        quote = api.get_quote(symbol_raw)
        api.wait_update()
        price_limit = quote.last_price
        if not price_limit>=0:
            return

        if pos_long_his > 0:
            api.insert_order(symbol=symbol_raw, direction="SELL", offset="CLOSE", limit_price=price_limit,
                             volume=pos_long_his)
            api.wait_update()
            print(symbol_raw, "平昨多完成")
        if pos_long_today > 0:
            api.insert_order(symbol=symbol_raw, direction="SELL", offset="CLOSETODAY", limit_price=price_limit,
                             volume=pos_long_today)
            api.wait_update()
            print(symbol_raw, "平今多完成")
        if pos_short_his > 0:
            api.insert_order(symbol=symbol_raw, direction="BUY", offset="CLOSE", limit_price=price_limit,
                             volume=pos_short_his)
            api.wait_update()
            print(symbol_raw, "平昨空完成")
        if pos_short_today > 0:
            api.insert_order(symbol=symbol_raw, direction="BUY", offset="CLOSETODAY", limit_price=price_limit,
                             volume=pos_short_today)
            api.wait_update()
            print(symbol_raw, "平今空完成")


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


def get_alive_order_list(direction):
    global api
    result_set = set()
    # api.insert_order(symbol="DCE.m2109", direction="BUY", offset="OPEN", volume=3, limit_price=3632)
    alive_order_list = api.get_order()
    api.wait_update()
    for oid, order_unit in alive_order_list.items():
        if order_unit.status == "ALIVE" and order_unit.direction == direction:
            result_set.add(order_unit.exchange_id + "."+order_unit.instrument_id)
    return result_set


def perform_close(alive_close_set):
    position_list = api.get_position()
    item_list = list(position_list.keys())
    for holding_item in item_list:
        holding_position = api.get_position(holding_item)
        profit = holding_position.float_profit
        pos_long_his = holding_position.pos_long_his
        pos_long_today = holding_position.pos_long_today
        pos_short_his = holding_position.pos_short_his
        pos_short_today = holding_position.pos_short_today
        close_order(holding_item, profit, pos_long_his, pos_long_today, pos_short_his, pos_short_today, alive_close_set)


if __name__ == '__main__':
    while True:
        api.wait_update()
        holding_list = get_all_holding_set()
        # perform open
        alive_open_list = get_alive_order_list("OPEN")
        with open(RawList.html_filepath, 'r') as fp:
            json_obj = json.load(fp)
        json_list = json_obj['data']
        for item in json_list:
            open_order_json(item, holding_set=holding_list, alive_open_set=alive_open_list)
        api.wait_update()

        # perform close
        alive_close_list = get_alive_order_list("CLOSE")
        perform_close(alive_close_list)

        time.sleep(5)
