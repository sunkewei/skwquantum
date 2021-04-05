# -*- coding: utf-8 -*-
import dash
import dash_html_components as html
import dash_core_components as dcc
import json


pinzhong_list_json = {
    'pinzhonglist':[
        {'name': '20号胶', 'code': 'NR2105'},
        {'name': 'PTA', 'code': 'TA2105'},
        {'name': 'PVC', 'code': 'V2105'},
        {'name': '苯乙烯', 'code': 'EB2105'},
        {'name': '玻璃', 'code': 'FG2105'},
        {'name': '菜籽粕', 'code': 'RM2105'},
        {'name': '菜油', 'code': 'OI2105'},
        {'name': '纯碱', 'code': 'SA2105'},
        {'name': '低硫油', 'code': 'LU2105'},
        {'name': '动力煤', 'code': 'ZC2105'},
        {'name': '豆粕', 'code': 'M2105'},
        {'name': '豆油', 'code': 'Y2105'},
        {'name': '硅铁', 'code': 'SF2105'},
        {'name': '鸡蛋', 'code': 'JD2105'},
        {'name': '甲醇', 'code': 'MA2105'},
        {'name': '焦煤', 'code': 'JM2105'},
        {'name': '焦炭', 'code': 'J2105'},
        {'name': '聚丙烯', 'code': 'PP2105'},
        {'name': '沥青', 'code': 'BU2106'},
        {'name': '螺纹钢', 'code': 'RB2105'},
        {'name': '锰硅', 'code': 'SM2105'},
        {'name': '棉花', 'code': 'CF2105'},
        {'name': '苹果', 'code': 'AP2105'},
        {'name': '热卷', 'code': 'HC2105'},
        {'name': '塑料', 'code': 'L2105'},
        {'name': '白糖', 'code': 'SR2105'},
        {'name': '铁矿', 'code': 'I2105'},
        {'name': '液化气', 'code': 'PG2106'},
        {'name': '乙二醇', 'code': 'EG2105'},
        {'name': '玉米', 'code': 'C2105'},
        {'name': '纸浆', 'code': 'SP2105'},
        {'name': '棕榈油', 'code': 'P2105'},
        {'name': '铜', 'code': 'CU2105'},
    ]
}


def get_pinzhong_list():
    result = []
    for pinzhong in pinzhong_list_json['pinzhonglist']:
        result.append(pinzhong)
    return result


def generate_tab_content(pinzhong):
    tab_style = {
        "background": "#323130",
        'text-transform': 'uppercase',
        'color': 'white',
        'border': 'grey',
        'font-size': '11px',
        'font-weight': 600,
        'align-items': 'center',
        'justify-content': 'center',
        'border-radius': '4px',
        'padding': '6px'
    }
    tab_selected_style = {
        "background": "grey",
        'text-transform': 'uppercase',
        'color': 'white',
        'font-size': '11px',
        'font-weight': 600,
        'align-items': 'center',
        'justify-content': 'center',
        'border-radius': '4px',
        'padding': '6px'
    }

    result = dcc.Tab(label=pinzhong['name'], children=[
        dcc.Graph(
            figure={
                'data': [
                    {'x': [1, 2, 3], 'y': [4, 1, 2],
                     'type': 'bar', 'name': 'SF'},
                    {'x': [1, 2, 3], 'y': [2, 4, 5],
                     'type': 'bar', 'name': u'Montréal'},
                ]
            }
        )
    ], style=tab_style, selected_style=tab_selected_style)
    return result

def generate_tab_list(pinzhong_list):
    result = []
    for pinzhong in pinzhong_list:
        pinzhong_tab = generate_tab_content(pinzhong)
        result.append(pinzhong_tab)
    return result


def generate_framework_div(pinzhong_list):
    tab_list = generate_tab_list(pinzhong_list)
    print(tab_list)
    return html.Div([dcc.Tabs(children=tab_list)])


#external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__)

pinzhong_list = get_pinzhong_list()



app.layout = generate_framework_div(pinzhong_list)


if __name__ == '__main__':
    app.run_server(debug=True)