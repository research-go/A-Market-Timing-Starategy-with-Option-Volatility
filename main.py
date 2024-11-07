import numpy as np
import pandas as pd
import openpyxl
import option
import datetime as dt
from tqdm import tqdm

'''
data = pd.read_csv('./data.csv')
iv_list = []
for i in range(len(data)):
    tmp = option.Option(data.iloc[i, 1], 400, 0.001929, 0.054795)
    iv_list.append(tmp.implied_volatility(data.iloc[i, 2])*100)
data['Implied Volatility'] = iv_list
data.to_csv('./result.csv')
'''

data_vkospi = pd.read_excel('vkospi.xlsx', usecols=[0, 1])
data_vkospi.drop([0, 1], inplace=True)
data_vkospi.columns = ['date', 'Last Price']
data_vkospi = data_vkospi.set_index('date')
data_put_call = pd.read_excel('put_call_ratio.xlsx')
data_put_call = data_put_call.set_index('date')
data_call_put_average_real = pd.read_excel('call_put_average_real.xlsx')
data_call_put_average_real = data_call_put_average_real.set_index('date')


def data_export(start, end):
    data = option.data_put
    start = dt.datetime.strptime(start, "%Y-%m-%d")
    end = dt.datetime.strptime(end, "%Y-%m-%d")
    result = pd.DataFrame(columns=['kospi200', 'call', 'put', 'aggregate', 'real_vkospi', 'put_call_ratio', 'call_put_average_real'])
    for i in tqdm(data.index):
        if start <= i <= end:
            tmp = option.Vkospi(i)
            row = {
                'kospi200': data.loc[i, 'kospi200'],
                'call': tmp.index(position='call'),
                'put': tmp.index(position='put'),
                'aggregate': tmp.index(position='both'),
                'real_vkospi': data_vkospi.loc[i, 'Last Price'],
                'put_call_ratio': data_put_call.loc[i, 'Last Price'],
                'call_put_average_real': data_call_put_average_real.loc[i, 'Last Price'],
            }
            result = pd.concat([result, pd.DataFrame(row, index=[i])])
    print(result)
    result.to_csv('put_call.csv')


def strategy(col='call_put_var_ratio', is_high=True, percentile=90):
    data = pd.read_csv('put_call.csv')
    data.columns = ['date', 'kospi200', 'call', 'put', 'aggregate', 'real_vkospi', 'put_call_ratio', 'call_put_average_real']
    data = data.set_index('date')
    data = data.sort_index()
    data['call_put_var_ratio'] = data['call']/data['put']
    data['kospi200_tomorrow'] = np.log(data['kospi200']/data['kospi200'].shift(1))*100
    data['kospi200_tomorrow'] = data['kospi200_tomorrow'].shift(-1)
    data.drop([data.index[len(data)-1]], inplace=True)
    percentile_list = []
    for i in range(len(data)):
        if i >= 252 - 1:
            percentile_list.append(np.nanpercentile(data[col][i - 251:i + 1], percentile))
        else:
            percentile_list.append(np.nan)
    data['percentile'] = percentile_list
    data.to_csv('result.csv')
    if is_high:
        result = data[data[col] >= data['percentile']]
    else:
        result = data[data[col] <= data['percentile']]

    average_return = sum(result['kospi200_tomorrow'])/len(result['kospi200_tomorrow'])
    win_ratio = len(result[result['kospi200_tomorrow'] > 0])/len(result['kospi200_tomorrow'])*100
    opportunities = len(result['kospi200_tomorrow'])/len(data)*252
    return average_return, win_ratio, opportunities


if __name__ == "__main__":
    # data_export("2019-01-03", "2024-07-30")
    result = pd.DataFrame(columns=['col', 'is_high', 'percentile(%)', 'Average Return'])
    for i in [99, 95, 90]:
        row = {
            'col': 'call_put_average_real',
            'is_high': True,
            'percentile(%)': i
        }
        tmp = strategy(col=row['col'], is_high=row['is_high'], percentile=row['percentile(%)'])
        row['Average Return'] = tmp[0]
        row['Win_Ratio(%)'] = tmp[1]
        row['1Y Opportunity'] = tmp[2]
        result = pd.concat([result, pd.DataFrame(row, index=[i])])
    for i in [10, 5, 1]:
        row = {
            'col': 'call_put_average_real',
            'is_high': False,
            'percentile(%)': i
        }
        tmp = strategy(col=row['col'], is_high=row['is_high'], percentile=row['percentile(%)'])
        row['Average Return'] = tmp[0]
        row['Win_Ratio(%)'] = tmp[1]
        row['1Y Opportunity'] = tmp[2]
        result = pd.concat([result, pd.DataFrame(row, index=[i])])
    for i in [99, 95, 90]:
        row = {
            'col': 'call_put_var_ratio',
            'is_high': True,
            'percentile(%)': i
        }
        tmp = strategy(col=row['col'], is_high=row['is_high'], percentile=row['percentile(%)'])
        row['Average Return'] = tmp[0]
        row['Win_Ratio(%)'] = tmp[1]
        row['1Y Opportunity'] = tmp[2]
        result = pd.concat([result, pd.DataFrame(row, index=[i])])
    for i in [10, 5, 1]:
        row = {
            'col': 'call_put_var_ratio',
            'is_high': False,
            'percentile(%)': i
        }
        tmp = strategy(col=row['col'], is_high=row['is_high'], percentile=row['percentile(%)'])
        row['Average Return'] = tmp[0]
        row['Win_Ratio(%)'] = tmp[1]
        row['1Y Opportunity'] = tmp[2]
        result = pd.concat([result, pd.DataFrame(row, index=[i])])
    result.to_csv('result.csv')



