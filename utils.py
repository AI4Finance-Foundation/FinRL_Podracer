from stockstats import StockDataFrame as Sdf
import alpaca_trade_api as tradeapi
import pandas as pd
import numpy as np

tech_indicator_list = ['macd', 'boll_ub', 'boll_lb', 'rsi_30', 'dx_30', 'cci_30',
                       'close_30_sma', 'close_60_sma']


def add_technical_indicator(data):
    df = data.copy()
    df = df.sort_values(by=['tic', 'date'])
    stock = Sdf.retype(df.copy())
    unique_ticker = stock.tic.unique()

    for indicator in tech_indicator_list:
        indicator_df = pd.DataFrame()
        for i in range(len(unique_ticker)):
            try:
                temp_indicator = stock[stock.tic == unique_ticker[i]][indicator]
                temp_indicator = pd.DataFrame(temp_indicator)
                temp_indicator['tic'] = unique_ticker[i]
                temp_indicator['date'] = df[df.tic == unique_ticker[i]]['date'].to_list()
                indicator_df = indicator_df.append(
                    temp_indicator, ignore_index=True
                )
            except Exception as e:
                print(e)
        df = df.merge(indicator_df[['tic', 'date', indicator]], on=['tic', 'date'], how='left')
    df = df.sort_values(by=['date', 'tic'])
    return df


def data_split(df, start, end):
    data = df[(df.date >= start) & (df.date < end)]
    data = data.sort_values(["date", "tic"], ignore_index=True)
    data.index = data.date.factorize()[0]
    return data


def data_fetch(API_KEY, API_SECRET, APCA_API_BASE_URL, stock_list
               , start_date='2021-05-10',
               end_date='2021-05-10', time_interval='15Min'):
    api = tradeapi.REST(API_KEY, API_SECRET, APCA_API_BASE_URL, 'v2')
    NY = 'America/New_York'
    start_date = pd.Timestamp(start_date, tz=NY)
    end_date = pd.Timestamp(end_date, tz=NY) + pd.Timedelta(days=1)
    date = start_date
    dataset = None
    if_first_time = True
    while date != end_date:
        start_time = (date + pd.Timedelta('09:30:00')).isoformat()
        end_time = (date + pd.Timedelta('16:00:00')).isoformat()
        print('Data before ' + end_time + ' is successfully fetched')
        barset = api.get_barset(stock_list, time_interval, start=start_time,
                                end=end_time, limit=500)
        if if_first_time:
            dataset = barset.df
            if_first_time = False
        else:
            dataset = dataset.append(barset.df)
        date = date + pd.Timedelta(days=1)
        if date.isoformat()[-14:-6] == '01:00:00':
            date = date - pd.Timedelta('01:00:00')
        elif date.isoformat()[-14:-6] == '23:00:00':
            date = date + pd.Timedelta('01:00:00')
        if date.isoformat()[-14:-6] != '00:00:00':
            raise ValueError('Timezone Error')

    return dataset


def preprocess(alpaca_df, stock_list):
    alpaca_df = alpaca_df.fillna(axis=0, method='ffill')
    alpaca_df = alpaca_df.fillna(axis=0, method='bfill')
    alpaca_df = alpaca_df.dropna()
    if_first_time = True
    for stock in stock_list:
        df = alpaca_df[stock]
        df = df.reset_index()
        n = df.shape[0]
        df['time_index'] = range(n)
        df['tic'] = stock
        ary = df.values
        if if_first_time:
            dataset = ary
            if_first_time = False
        else:
            dataset = np.vstack((dataset, ary))
    return dataset


