import gym
from elegantrl.env import PreprocessEnv
from elegantrl.run import Arguments, train_and_evaluate, train_and_evaluate_mp


def demo3_custom_env_fin_rl():
    from elegantrl.agent import AgentPPO

    '''choose an DRL algorithm'''
    args = Arguments(if_on_policy=True)
    args.agent = AgentPPO()
    args.agent.if_use_gae = False

    "TotalStep:  5e4, TargetReturn: 1.25, UsedTime:  20s, FinanceStock-v2"
    "TotalStep: 20e4, TargetReturn: 1.50, UsedTime:  80s, FinanceStock-v2"
    # from elegantrl.env import FinanceStockEnv  # a standard env for ElegantRL, not need PreprocessEnv()
    # args.env = FinanceStockEnv(if_train=True, train_beg=0, train_len=1024)
    # args.env_eval = FinanceStockEnv(if_train=False, train_beg=0, train_len=1024)  # eva_len = 1699 - train_len
    from FinRL import StockTradingEnv
    from finrl.config import config
    env_kwargs = {
        "max_stock": 100,
        "initial_amount": 1000000,
        "buy_cost_pct": 0.001,
        "sell_cost_pct": 0.001,
        # "state_space": 1 + (2 + len(config.TECHNICAL_INDICATORS_LIST)) * stock_dimension,
        "stock_dim": len(config.DOW_30_TICKER),
        "tech_indicator_list": config.TECHNICAL_INDICATORS_LIST,
        # "action_space": stock_dimension,
        "reward_scaling": 1e-4
    }
    args.env = StockTradingEnv(**env_kwargs)

    args.reward_scale = 2 ** 0  # RewardRange: 0 < 1.0 < 1.25 < 1.5 < 1.6
    args.break_step = int(5e6)
    args.net_dim = 2 ** 8
    args.max_step = args.env.max_step
    args.max_memo = (args.max_step - 1) * 8
    args.batch_size = 2 ** 11
    args.repeat_times = 2 ** 4
    args.eval_times1 = 2 ** 1
    args.eval_times2 = 2 ** 3
    args.if_allow_break = True

    '''train and evaluate'''
    # train_and_evaluate(args)
    args.rollout_num = 4
    train_and_evaluate_mp(args)


if __name__ == '__main__':
    demo3_custom_env_fin_rl()
