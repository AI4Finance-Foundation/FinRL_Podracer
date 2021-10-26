import sys
import gym
from FinRLPodracer.elegantrl.run import Arguments, train_and_evaluate_mp, train_and_evaluate_mg

gym.logger.set_level(40)  # Block warning: 'WARN: Box bound precision lowered by casting to float32'


def demo_custom_env_finance_rl():
    args = Arguments(if_on_policy=True)  # hyper-parameters of on-policy is different from off-policy
    args.random_seed = 0

    from FinRLPodracer.elegantrl.agent import AgentPPO
    args.agent = AgentPPO()
    args.agent.cri_target = True
    args.agent.lambda_entropy = 0.04

    from FinRLPodracer.FinRL.StockTrading import StockEnvNAS89
    args.gamma = 0.999
    args.env = StockEnvNAS89(if_eval=False, gamma=args.gamma)
    args.eval_env = StockEnvNAS89(if_eval=True, gamma=args.gamma)

    args.repeat_times = 2 ** 4
    args.learning_rate = 2 ** -14
    args.net_dim = int(2 ** 8 * 1.5)
    args.batch_size = args.net_dim * 4

    if_single_env = 0
    if if_single_env:
        args.gpu_id = 0
        args.worker_num = 4
        train_and_evaluate_mp(args)

    if_multi_learner = 1
    if if_multi_learner:
        args.env = StockEnvNAS89(if_eval=False, gamma=args.gamma)
        args.gpu_id = (0, 1)
        args.worker_num = 2
        train_and_evaluate_mg(args)


def demo_custom_env_finance_rl_dow30():  # 1.7+ 2.0+
    args = Arguments(if_on_policy=True)  # hyper-parameters of on-policy is different from off-policy
    args.random_seed = 1943

    from FinRLPodracer.elegantrl.agent import AgentPPO
    args.agent = AgentPPO()
    args.agent.cri_target = True
    args.agent.lambda_entropy = 0.02

    args.gamma = 0.995

    from FinRLPodracer.FinRL.StockTrading import StockEnvDOW30
    args.env = StockEnvDOW30(if_eval=False, gamma=args.gamma)
    args.eval_env = StockEnvDOW30(if_eval=True, gamma=args.gamma)

    args.repeat_times = 2 ** 4
    args.learning_rate = 2 ** -14
    args.net_dim = 2 ** 8
    args.batch_size = args.net_dim * 2

    args.eval_gap = 2 ** 7
    args.eval_times1 = 2 ** 0
    args.eval_times2 = 2 ** 2
    args.break_step = int(5e6)  # int(args.env.max_step * 2000)
    args.if_allow_break = False

    if_single_env = 0
    if if_single_env:
        args.gpu_id = int(sys.argv[-1][-4])
        args.random_seed += int(args.gpu_id)
        args.target_step = args.env.max_step * 4
        args.worker_num = 4
        train_and_evaluate_mp(args)

    if_multi_learner = 0
    if if_multi_learner:
        args.env = StockEnvDOW30(if_eval=False, gamma=args.gamma)
        args.gpu_id = (0, 1)
        args.worker_num = 2
        train_and_evaluate_mg(args)


def demo_custom_env_finance_rl_nas74():  # 1.7+ 2.0+
    args = Arguments(if_on_policy=True)  # hyper-parameters of on-policy is different from off-policy
    args.random_seed = 1943

    from FinRLPodracer.elegantrl.agent import AgentPPO
    args.agent = AgentPPO()
    args.agent.cri_target = True
    args.agent.lambda_entropy = 0.02

    # args.gamma = 0.999 # ebta
    args.gamma = 0.997  # ceta

    from FinRLPodracer.FinRL.StockTrading import StockEnvNAS74
    args.env = StockEnvNAS74(if_eval=False, gamma=args.gamma, turbulence_thresh=700)
    args.eval_env = StockEnvNAS74(if_eval=True, gamma=args.gamma, turbulence_thresh=700)

    args.net_dim = 2 ** 8
    args.repeat_times = 2 ** 4
    args.learning_rate = 2 ** -14
    args.batch_size = args.net_dim * 4

    args.eval_gap = 2 ** 8
    args.eval_times1 = 2 ** 0
    args.eval_times2 = 2 ** 1
    args.break_step = int(25e6)
    args.if_allow_break = False

    if_single_env = 1
    if if_single_env:
        args.gpu_id = int(sys.argv[-1][-4])
        args.random_seed += int(args.gpu_id)
        args.target_step = args.env.max_step * 1
        args.worker_num = 8
        train_and_evaluate_mp(args)

    if_multi_learner = 0
    if if_multi_learner:
        args.gpu_id = (2, 3) if len(sys.argv) == 1 else eval(sys.argv[-1])  # python main.py -GPU 0,1
        args.repeat_times = 2 ** 4
        args.target_step = args.env.max_step
        args.worker_num = 4
        train_and_evaluate_mg(args)


def demo_custom_env_finance_rl_nas89():  # 1.7+ 2.0+
    args = Arguments(if_on_policy=True)  # hyper-parameters of on-policy is different from off-policy
    args.random_seed = 19430

    from FinRLPodracer.elegantrl.agent import AgentPPO
    args.agent = AgentPPO()
    args.agent.lambda_entropy = 0.02

    from FinRLPodracer.FinRL.StockTrading import StockEnvNAS89
    args.gamma = 0.999
    args.env = StockEnvNAS89(if_eval=False, gamma=args.gamma, turbulence_thresh=30)
    args.eval_env = StockEnvNAS89(if_eval=True, gamma=args.gamma, turbulence_thresh=15)

    args.net_dim = 2 ** 9
    args.repeat_times = 2 ** 4
    args.learning_rate = 2 ** -14
    args.batch_size = args.net_dim * 4

    args.eval_gap = 2 ** 8
    args.eval_times1 = 2 ** 0
    args.eval_times2 = 2 ** 1
    args.break_step = int(8e6)
    args.if_allow_break = False

    if_single_env = 1
    if if_single_env:
        args.gpu_id = int(sys.argv[-1][-4])
        args.random_seed += int(args.gpu_id)
        args.target_step = args.env.max_step * 1
        args.worker_num = 4
        train_and_evaluate_mp(args)

    if_multi_learner = 0
    if if_multi_learner:
        args.gpu_id = (2, 3) if len(sys.argv) == 1 else eval(sys.argv[-1])  # python main.py -GPU 0,1
        args.repeat_times = 2 ** 4
        args.target_step = args.env.max_step
        args.worker_num = 4
        train_and_evaluate_mg(args)


if __name__ == '__main__':
    demo_custom_env_finance_rl()
    # demo_custom_env_finance_rl_dow30()
    # demo_custom_env_finance_rl_nas74()
    # demo_custom_env_finance_rl_nas89()
