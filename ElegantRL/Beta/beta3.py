import os
import pandas as pd
import numpy as np
import numpy.random as rd


class StockTradingEnv:
    def __init__(self,
                 df,
                 stock_dim,
                 max_stock,
                 initial_amount,
                 buy_cost_pct,
                 sell_cost_pct,
                 reward_scaling,
                 state_space,
                 action_space,
                 tech_indicator_list,
                 turbulence_threshold=None,
                 make_plots=False,
                 print_verbosity=10,
                 day=0,
                 initial=True,
                 previous_state=None,
                 model_name='',
                 mode='',
                 iteration=''):
        self.day = day
        self.df = df
        self.stock_dim = stock_dim
        self.hmax = max_stock
        self.initial_amount = initial_amount
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.reward_scaling = reward_scaling
        self.state_space = state_space
        self.action_space = action_space
        self.tech_indicator_list = tech_indicator_list
        self.data = self.df.loc[self.day, :]
        self.terminal = False
        self.make_plots = make_plots
        self.print_verbosity = print_verbosity
        self.turbulence_threshold = turbulence_threshold
        self.initial = initial
        self.previous_state = list() if previous_state is None else previous_state
        self.model_name = model_name
        self.mode = mode
        self.iteration = iteration
        # initalize state
        self.state = self._initiate_state()

        # initialize reward
        self.reward = 0
        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        self.episode = 0

        # environment information
        self.env_name = 'StockTradingEnv-v1'
        self.action_dim = action_space
        self.state_dim = state_space
        self.max_step = len(self.df.index.unique()) - 1
        self.if_discrete = False
        self.target_return = 1234.0

    def reset(self):
        # initiate state
        self.state = self._initiate_state()

        self.day = 0
        self.data = self.df.loc[self.day, :]
        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        self.terminal = False

        self.episode += 1

        return np.array(self.state, dtype=np.float32)

    def step(self, actions):
        self.terminal = self.day >= len(self.df.index.unique()) - 1
        if self.terminal:
            state = np.array(self.state, dtype=np.float32)
            return state, self.reward, self.terminal, {}

        else:
            actions = actions * self.hmax  # actions initially is scaled between 0 to 1
            actions = (actions.astype(int))  # convert into integer because we can't by fraction of shares
            if self.turbulence_threshold is not None:
                if self.turbulence >= self.turbulence_threshold:
                    actions = np.array([-self.hmax] * self.stock_dim)
            begin_total_asset = self.state[0] + (
                    np.array(self.state[1:(self.stock_dim + 1)]) *
                    np.array(self.state[(self.stock_dim + 1):(self.stock_dim * 2 + 1)])).sum()
            # print("begin_total_asset:{}".format(begin_total_asset))

            sell_index = np.where(actions < 0)[0]
            buy_index = np.where(actions > 0)[0]

            for index in sell_index:
                # print(f"Num shares before: {self.state[index+self.stock_dim+1]}")
                # print(f'take sell action before : {actions[index]}')
                actions[index] = self._sell_stock(index, actions[index]) * (-1)
                # print(f'take sell action after : {actions[index]}')
                # print(f"Num shares after: {self.state[index+self.stock_dim+1]}")

            for index in buy_index:
                # print('take buy action: {}'.format(actions[index]))
                actions[index] = self._buy_stock(index, actions[index])

            self.day += 1
            self.data = self.df.loc[self.day, :]
            if self.turbulence_threshold is not None:
                self.turbulence = self.data['turbulence'].values[0]
            self.state = self._update_state()

            end_total_asset = self.state[0] + (
                    np.array(self.state[1:(self.stock_dim + 1)]) *
                    np.array(self.state[(self.stock_dim + 1):(self.stock_dim * 2 + 1)])).sum()
            self.reward = end_total_asset - begin_total_asset
            self.reward = self.reward * self.reward_scaling

        state = np.array(self.state, dtype=np.float32)
        return state, self.reward, self.terminal, {}

    def _sell_stock(self, index, action):
        if self.state[index + 1] > 0:
            # Sell only if the price is > 0 (no missing data in this particular date)
            # perform sell action based on the sign of the action
            if self.state[index + self.stock_dim + 1] > 0:
                # Sell only if current asset is > 0
                sell_num_shares = min(abs(action), self.state[index + self.stock_dim + 1])
                sell_amount = self.state[index + 1] * sell_num_shares * (1 - self.sell_cost_pct)
                # update balance
                self.state[0] += sell_amount

                self.state[index + self.stock_dim + 1] -= sell_num_shares
                self.cost += self.state[index + 1] * sell_num_shares * self.sell_cost_pct
                self.trades += 1
            else:
                sell_num_shares = 0
        else:
            sell_num_shares = 0

        return sell_num_shares

    def _buy_stock(self, index, action):
        if self.state[index + 1] > 0:
            # Buy only if the price is > 0 (no missing data in this particular date)
            available_amount = self.state[0] // self.state[index + 1]
            # print('available_amount:{}'.format(available_amount))

            # update balance
            buy_num_shares = min(available_amount, action)
            buy_amount = self.state[index + 1] * buy_num_shares * (1 + self.buy_cost_pct)
            self.state[0] -= buy_amount

            self.state[index + self.stock_dim + 1] += buy_num_shares

            self.cost += self.state[index + 1] * buy_num_shares * self.buy_cost_pct
            self.trades += 1
        else:
            buy_num_shares = 0

        return buy_num_shares

    def _initiate_state(self):
        if self.initial:
            # For Initial State
            if len(self.df.tic.unique()) > 1:
                # for multiple stock
                state = [self.initial_amount] + \
                        self.data.close.values.tolist() + \
                        [0] * self.stock_dim + \
                        sum([self.data[tech].values.tolist() for tech in self.tech_indicator_list], [])
            else:
                # for single stock
                state = [self.initial_amount] + \
                        [self.data.close] + \
                        [0] * self.stock_dim + \
                        sum([[self.data[tech]] for tech in self.tech_indicator_list], [])
        else:
            # Using Previous State
            if len(self.df.tic.unique()) > 1:
                # for multiple stock
                state = [self.previous_state[0]] + \
                        self.data.close.values.tolist() + \
                        self.previous_state[(self.stock_dim + 1):(self.stock_dim * 2 + 1)] + \
                        sum([self.data[tech].values.tolist() for tech in self.tech_indicator_list], [])
            else:
                # for single stock
                state = [self.previous_state[0]] + \
                        [self.data.close] + \
                        self.previous_state[(self.stock_dim + 1):(self.stock_dim * 2 + 1)] + \
                        sum([[self.data[tech]] for tech in self.tech_indicator_list], [])
        return state

    def _update_state(self):
        if len(self.df.tic.unique()) > 1:
            # for multiple stock
            state = [self.state[0]] + \
                    self.data.close.values.tolist() + \
                    list(self.state[(self.stock_dim + 1):(self.stock_dim * 2 + 1)]) + \
                    sum([self.data[tech].values.tolist() for tech in self.tech_indicator_list], [])

        else:
            # for single stock
            state = [self.state[0]] + \
                    [self.data.close] + \
                    list(self.state[(self.stock_dim + 1):(self.stock_dim * 2 + 1)]) + \
                    sum([[self.data[tech]] for tech in self.tech_indicator_list], [])

        return state

    def _get_date(self):
        if len(self.df.tic.unique()) > 1:
            date = self.data.date.unique()[0]
        else:
            date = self.data.date
        return date


def load_stock_trading_data():
    from finrl.config import config

    cwd = './env/FinRL'
    raw_data_path = f'{cwd}/StockTradingEnv_raw_data.df'
    processed_data_path = f'{cwd}/StockTradingEnv_processed_data.df'

    os.makedirs(cwd, exist_ok=True)

    print("==============Start Fetching Data===========")
    if os.path.exists(raw_data_path):
        raw_df = pd.read_pickle(raw_data_path)  # DataFrame of Pandas
        print('| raw_df.columns.values:', raw_df.columns.values)
    else:
        from finrl.marketdata.yahoodownloader import YahooDownloader
        raw_df = YahooDownloader(
            start_date=config.START_DATE,
            end_date=config.END_DATE,
            ticker_list=config.DOW_30_TICKER,
        ).fetch_data()
        raw_df.to_pickle(raw_data_path)

    print("==============Start Feature Engineering===========")
    if os.path.exists(processed_data_path):
        processed_df = pd.read_pickle(processed_data_path)  # DataFrame of Pandas
        print('| processed_df.columns.values:', processed_df.columns.values)
    else:
        from finrl.preprocessing.preprocessors import FeatureEngineer
        fe = FeatureEngineer(
            use_technical_indicator=True,
            tech_indicator_list=config.TECHNICAL_INDICATORS_LIST,
            use_turbulence=True,
            user_defined_feature=False,
        )
        processed_df = fe.preprocess_data(raw_df)
        processed_df.to_pickle(processed_data_path)

    # Training & Trading data split
    from finrl.preprocessing.data import data_split
    train_df = data_split(processed_df, '2008-03-19', '2016-01-01')  # 1963/3223
    eval_df = data_split(processed_df, '2016-01-01', '2021-01-01')  # 1260/3223

    return train_df, eval_df


def check_stock_env():
    from finrl.config import config
    train_df, eval_df = load_stock_trading_data()
    # train = data_split(processed_df, config.START_DATE, config.START_TRADE_DATE)
    # trade = data_split(processed_df, config.START_TRADE_DATE, config.END_DATE)

    # calculate state action space
    stock_dimension = len(train_df.tic.unique())
    state_space = 1 + (2 + len(config.TECHNICAL_INDICATORS_LIST)) * stock_dimension

    env_kwargs = {"hmax": 100,
                  "initial_amount": 1000000,
                  "buy_cost_pct": 0.001,
                  "sell_cost_pct": 0.001,
                  "state_space": state_space,
                  "stock_dim": stock_dimension,
                  "tech_indicator_list": config.TECHNICAL_INDICATORS_LIST,
                  "action_space": stock_dimension,
                  "reward_scaling": 2 ** -14}

    env = StockTradingEnv(df=train_df, **env_kwargs)
    action_dim = env.action_space

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
        print(';', len(next_state), env.day, reward)
        step += 1

    print(f"step: {step}, UsedTime: {time() - timer:.3f}")  # 44 seconds
    print(f"terminal reward {reward:.3f}")  # 44 seconds
    # print(f"episode return {env.episode_return:.3f}")  # 44 seconds


if __name__ == '__main__':
    check_stock_env()
