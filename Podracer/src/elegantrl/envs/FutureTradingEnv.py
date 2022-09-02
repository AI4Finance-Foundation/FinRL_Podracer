import os
import numpy as np
import pandas as pd
import numpy.random as rd
import matplotlib.pyplot as plt


class FutureTradingEnv:
    def __init__(self):
        data_dir = "./pro_data"

        self.step_per_day = int(240 * (60 / 15))  # 240 min per day, 60 seconds per minute, 15 seconds per step
        self.look_back_step = int(self.step_per_day * 16)
        self.max_step = int(self.step_per_day * 30)  # trading for 30 days
        self.max_position = 100  # max future position
        self.max_action = 10  # max future trading action

        self.total_price, self.total_time = self.convert_csv_to_data_frame_to_array(data_dir)

        end_sign_len = self.step_per_day * 5
        self.ary_end_sign = np.zeros(self.max_step, dtype=np.float32)
        self.ary_end_sign[-end_sign_len:] = np.linspace(0, 1, end_sign_len)

        '''reset'''
        self.idx = 0
        self.ary_price = None
        self.ary_time = None
        self.account = 0.0
        self.position = 0

    def reset(self):
        ary_len = self.ary_price.shape[0]
        i0 = rd.randint(self.look_back_step, ary_len - self.max_step)
        i1 = i0 + self.max_step
        self.ary_price = self.total_price[i0:i1]
        self.ary_time = self.total_time[i0:i1]

        self.idx = self.look_back_step
        return self.get_state()

    def step(self):
        pass

    def get_state(self):
        price = self.ary_price[self.idx]
        time = self.ary_time[self.idx]
        end_sign = self.ary_end_sign[self.idx]

        state = np.array((self.account, self.position, price, time, end_sign), dtype=np.float32)
        return state

    @staticmethod
    def convert_csv_to_data_frame_to_array(data_dir="./pro_data"):
        name = os.listdir(data_dir)[0]

        data = pd.read_csv(f"{data_dir}/{name}")
        data.dropna(inplace=True)  # drop the NAN line in data
        data.index = pd.DatetimeIndex(data.index)  # sort the data
        """
        ts: timestamp
        px: price
        date: year-month-date
        
                                 ts     px        date
        0       2018-01-02 09:30:15  2.793  2018-01-02
        1       2018-01-02 09:30:30  1.544  2018-01-02
        2       2018-01-02 09:30:45  1.130  2018-01-02
        3       2018-01-02 09:31:00  0.990  2018-01-02
        4       2018-01-02 09:31:15  0.992  2018-01-02
                                ...    ...         ...
        916795  2021-12-09 14:59:00  3.450  2021-12-09
        916796  2021-12-09 14:59:15  3.439  2021-12-09
        916797  2021-12-09 14:59:30  3.364  2021-12-09
        916798  2021-12-09 14:59:45  3.209  2021-12-09
        916799  2021-12-09 15:00:00  3.321  2021-12-09
        [916800 rows x 3 columns]
        """

        from datetime import datetime
        ary_timestamp = [datetime.fromisoformat(date_str).timestamp()
                         for date_str in data['ts'].to_list()]
        ary_time = np.array(ary_timestamp, dtype=np.float32)
        ary_time = ary_time + 3600 * 8  # '1970-01-02 08:00:00' --> timestamp=0
        ary_time = ary_time / (3600 * 24)  # delta of a single day is 3600*24
        ary_time = ary_time % 1.0 - 0.5  # mod a single day and do normalization

        ary_price = data['px'].to_numpy()
        # ary_price.shape = (-1, )
        # ary_time.shape = (-1, )
        return ary_price, ary_time


def demo__convert_timestamp_to_date():
    from datetime import datetime

    timestamp = 1234567890.0
    date_str = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
    print(f"timestamp {timestamp} --> date_str {date_str}")

    date_str = '1970-01-02 08:00:00'
    timestamp = datetime.fromisoformat(date_str).timestamp()
    print(f"date_str {date_str} --> timestamp {timestamp}")

    t1 = datetime.fromisoformat('2000-01-01 00:00:00').timestamp()
    t2 = datetime.fromisoformat('2000-01-02 00:00:00').timestamp()
    print(f"delta second of a single day {t2 - t1} == 3600*24")
    print(f"{t2 / 24 / 3600}")


if __name__ == '__main__':
    # check_compound_env()
    # check_future_trading_env()
    # demo__convert_timestamp_to_date()
    env = FutureTradingEnv()
