from multiprocessing import Process, Queue
from utils import data_fetch, data_split, add_technical_indicator, preprocess

API_KEY = "PK9ZDUS6Z4IPK2TMMJ1M"
API_SECRET = "sKTlBoUCyORn4g0Ju1xHE3iJXlvwdoB7awatNbtq"
APCA_API_BASE_URL = "https://paper-api.alpaca.markets"
stock_list = [
    "AAPL",
    "MSFT",
    "JPM",
    "V",
    "RTX",
    "PG",
    "GS",
    "NKE",
    "DIS",
    "AXP",
    "HD",
    "INTC",
    "WMT",
    "IBM",
    "MRK",
    "UNH",
    "KO",
    "CAT",
    "TRV",
    "JNJ",
    "CVX",
    "MCD",
    "VZ",
    "CSCO",
    "XOM",
    "BA",
    "MMM",
    "PFE",
    "WBA",
    "DD",
]
start = '2021-05-01'
end = '2021-05-10'
time_interval = '1Min'


def fetch(stock, index):
    alpaca_df = data_fetch(API_KEY, API_SECRET, APCA_API_BASE_URL, stock, start,
                           end, time_interval)
    alpaca_df.to_csv(f'./data/test_{stock}_{index}.csv')


if __name__ == "__main__":
    queue = Queue()
    print(len(stock_list))
    processes = [Process(target=fetch, args=(stock_list[i], i)) for i in range(len(stock_list))]

    for p in processes:
        p.start()

    for p in processes:
        p.join()
