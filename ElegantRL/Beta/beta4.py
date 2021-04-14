import os
import numpy as np
import numpy.random as rd


class StockTradingEnv:
    def __init__(self, max_stock=1e2, initial_amount=1e6, buy_cost_pct=1e-3, sell_cost_pct=1e-3,
                 if_train=True):
        # load data
        self.price_ary, self.tech_ary = self.get_price_ary_tech_ary()
        if if_train:
            self.price_ary, self.tech_ary = self.price_ary[:1932], self.tech_ary[:1932]
        else:
            self.price_ary, self.tech_ary = self.price_ary[1932:], self.tech_ary[1932:]
        max_day = self.price_ary.shape[0]
        stock_num = self.price_ary.shape[1]

        self.max_stock = max_stock
        self.initial_amount = initial_amount
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.initial_amount = initial_amount
        self.initial_stocks = np.zeros(stock_num, dtype=np.float32)
        self.initial_asset = None
        self.gamma = 0.99

        # reset()
        self.day = None
        self.rewards = None
        self.total_asset = None
        self.episode_return = 0

        self.amount = None
        self.stocks = None

        # environment information
        self.env_name = 'StockTradingEnv-v1'
        self.max_step = max_day
        self.action_dim = stock_num
        self.state_dim = len(self.reset())
        self.if_discrete = False
        self.target_return = 1234.0

    def reset(self):
        self.day = 0
        self.rewards = 0
        self.amount = self.initial_amount
        self.stocks = self.initial_stocks

        self.total_asset = (self.price_ary[self.day] * self.stocks).sum() + self.amount
        self.initial_asset = self.total_asset

        price_ary = self.price_ary[self.day]
        tech_ary = self.tech_ary[self.day]
        state = np.array((self.amount * 2 ** -16,
                          *(self.stocks * 2 ** -8),
                          *(price_ary * 2 ** -12),
                          *(tech_ary * 2 ** -8),), dtype=np.float32)
        return state

    def step(self, action):
        self.day += 1
        done = self.day == self.max_step - 1

        # actions initially is scaled between 0 to 1, convert into integer because we can't by fraction of shares
        # positive action denotes bug
        # negative action denotes sell
        action = ((action * self.max_stock).astype(int))

        price_ary = self.price_ary[self.day]
        for i in np.where(action < 0)[0]:  # sell_index
            available_stock = self.stocks[i]
            sell_num_shares = min(-action[i], available_stock)
            self.stocks[i] -= sell_num_shares

            sell_amount = price_ary[i] * sell_num_shares * (1 - self.sell_cost_pct)
            self.amount += sell_amount

        for i in np.where(action > 0)[0]:  # buy_index
            available_amount = self.amount // price_ary[i]
            buy_num_shares = min(action[i], available_amount)
            self.stocks[i] += buy_num_shares

            buy_amount = price_ary[i] * buy_num_shares * (1 + self.buy_cost_pct)
            self.amount -= buy_amount

        tech_ary = self.tech_ary[self.day]
        state = np.array((self.amount * 2 ** -16,
                          *(self.stocks * 2 ** -8),
                          *(price_ary * 2 ** -12),
                          *(tech_ary * 2 ** -8),), dtype=np.float32)

        total_asset = (price_ary * self.stocks).sum() + self.amount
        reward = (total_asset - self.total_asset) * 2 ** -16
        self.total_asset = total_asset

        self.rewards += reward
        if done:
            reward += 1 / (1 - self.gamma) * (self.rewards / self.max_step)  # todo
            self.episode_return = total_asset / self.initial_asset

        return state, reward, done, dict()

    def get_price_ary_tech_ary(self, ticker_list=None, tech_id_list=None, beg_date=None, end_date=None, ):
        """source: https://github.com/AI4Finance-LLC/FinRL-Library
        finrl/autotrain/training.py
        finrl/preprocessing/preprocessing.py
        finrl/env/env_stocktrading.py
        """

        """hyper-parameters"""
        cwd = './env/FinRL'
        ary_data_path = f'{cwd}/StockTradingEnv_ary_data.npz'
        raw_data_path = f'{cwd}/StockTradingEnv_raw_data.df'
        prp_data_path = f'{cwd}/StockTradingEnv_prp_data.df'  # preprocessed data
        beg_date = '2008-03-19' if beg_date is None else beg_date
        end_date = '2021-01-01' if end_date is None else end_date
        ticker_list = ['AAPL', 'MSFT', 'JPM', 'V', 'RTX', 'PG', 'GS', 'NKE', 'DIS', 'AXP', 'HD',
                       'INTC', 'WMT', 'IBM', 'MRK', 'UNH', 'KO', 'CAT', 'TRV', 'JNJ', 'CVX', 'MCD',
                       'VZ', 'CSCO', 'XOM', 'BA', 'MMM', 'PFE', 'WBA', 'DD'
                       ] if ticker_list is None else ticker_list
        tech_id_list = ['macd', 'boll_ub', 'boll_lb', 'rsi_30', 'cci_30', 'dx_30',
                        'close_30_sma', 'close_60_sma'
                        ] if tech_id_list is None else tech_id_list

        """load from *.npz file"""
        if not os.path.exists(ary_data_path):
            print(f"| FileNotFound: {ary_data_path}, so we download data from Internet.")
            print(f"  Can you download from github ↓ and put it in {ary_data_path}?")
            print(f"  https://github.com/Yonv1943/ElegantRL/env/FinRL/StockTradingEnv_ary_data.npz")
            print(f"  Or install finrl `pip3 install git+https://github.com/AI4Finance-LLC/FinRL-Library.git`")
            print(f"  Then it will download raw data {raw_data_path} from YaHoo")

            os.makedirs(cwd, exist_ok=True)
            input(f'| If you have downloaded *.npz file or install finrl, press ENTER:')

        if os.path.exists(ary_data_path):
            ary_dict = np.load(ary_data_path, allow_pickle=True)
            price_ary = ary_dict['price_ary']
            tech_ary = ary_dict['tech_ary']
            return price_ary, tech_ary

        '''download and generate *.npz when FileNotFound'''
        print(f"| get_close_ary_tech_ary(), load: {raw_data_path}")
        df = self.raw_data_download(raw_data_path, beg_date, end_date, ticker_list)
        print(f"| get_close_ary_tech_ary(), load: {ary_data_path}")
        df = self.raw_data_preprocess(prp_data_path, df, beg_date, end_date, tech_id_list, )
        # import pandas as pd
        # df = pd.read_pickle(prp_data_path)  # DataFrame of Pandas

        # convert part of DataFrame to Numpy
        tech_ary = list()
        price_ary = list()
        df_len = len(df.index.unique())
        print(df_len)
        from tqdm import trange
        for day in trange(df_len):
            item = df.loc[day]

            tech_items = [item[tech].values.tolist() for tech in tech_id_list]
            tech_items_flatten = sum(tech_items, [])
            tech_ary.append(tech_items_flatten)

            price_ary.append(item.close)

        price_ary = np.array(price_ary)
        tech_ary = np.array(tech_ary)
        print(f"| get_close_ary_tech_ary, price_ary.shape: {price_ary.shape}")
        print(f"| get_close_ary_tech_ary, tech_ary.shape: {tech_ary.shape}")
        np.savez_compressed(ary_data_path,
                            close_ary=np.array(price_ary),
                            tech_ary=np.array(tech_ary))
        return price_ary, tech_ary

    @staticmethod
    def raw_data_download(raw_data_path, beg_date, end_date, ticker_list):
        if os.path.exists(raw_data_path):
            import pandas as pd
            raw_df = pd.read_pickle(raw_data_path)  # DataFrame of Pandas
            print('| raw_df.columns.values:', raw_df.columns.values)
            # ['date' 'open' 'high' 'low' 'close' 'volume' 'tic' 'day']
        else:
            from finrl.marketdata.yahoodownloader import YahooDownloader
            yd = YahooDownloader(start_date=beg_date, end_date=end_date, ticker_list=ticker_list, )
            raw_df = yd.fetch_data()
            raw_df.to_pickle(raw_data_path)
        return raw_df

    @staticmethod
    def raw_data_preprocess(prp_data_path, df, beg_date, end_date, tech_id_list, ):
        if os.path.exists(prp_data_path):
            import pandas as pd
            df = pd.read_pickle(prp_data_path)  # DataFrame of Pandas
        else:
            from finrl.preprocessing.preprocessors import FeatureEngineer
            fe = FeatureEngineer(use_technical_indicator=True, tech_indicator_list=tech_id_list,
                                 use_turbulence=True, user_defined_feature=False, )
            df = fe.preprocess_data(df)  # preprocess raw_df

            df = df[(df.date >= beg_date) & (df.date < end_date)]
            df = df.sort_values(["date", "tic"], ignore_index=True)
            df.index = df.date.factorize()[0]

            df.to_pickle(prp_data_path)

        print('| df.columns.values:', df.columns.values)
        assert all(df.columns.values == [
            'date', 'open', 'high', 'low', 'close', 'volume', 'tic', 'day', 'macd',
            'boll_ub', 'boll_lb', 'rsi_30', 'cci_30', 'dx_30', 'close_30_sma',
            'close_60_sma', 'turbulence'])
        return df


def check_stock_trading_env():
    env = StockTradingEnv()
    action_dim = env.action_dim

    state = env.reset()
    print('state_dim', len(state))

    done = False
    step = 1
    reward = None
    from time import time
    timer = time()
    while not done:
        action = rd.rand(action_dim) * 2 - 1
        next_state, reward, done, _ = env.step(action)
        print(';', next_state.shape, env.day, reward)
        step += 1

    print(f"step: {step}, UsedTime: {time() - timer:.3f}")  # 44 seconds
    print(f"episode return {env.episode_return:.3f}")  # 44 seconds
    print(f"terminal reward {reward:.3f}")  # 44 seconds


if __name__ == '__main__':
    # check_finance_stock_env()
    check_stock_trading_env()
