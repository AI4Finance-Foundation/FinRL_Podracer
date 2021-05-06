from elegant_finrl.run import *
from elegant_finrl.agent import AgentPPO, AgentDDPG
from elegant_finrl.env import StockTradingEnv
import yfinance as yf
from stockstats import StockDataFrame as Sdf

args = Arguments(if_on_policy=True)
args.agent = AgentPPO()
args.env = StockTradingEnv(cwd='./', if_eval=True)
args.if_remove = False
args.cwd = './AgentPPO/StockTradingEnv-v1_0'
args.init_before_training()

prediction = args.env.trade_prediction(args, torch)

args.env.backtest_plot(prediction, baseline_ticker = '^DJI', baseline_start = '2019-01-01', baseline_end = '2021-01-01')
