import sys
import gym

IF_GA = True  # todo
if IF_GA:
    from FinRLPodracer.elegantrl.run_ga import Arguments, train_and_evaluate_mp, train_and_evaluate_mg
else:
    from FinRLPodracer.elegantrl.run import Arguments, train_and_evaluate_mp, train_and_evaluate_mg

gym.logger.set_level(40)  # Block warning: 'WARN: Box bound precision lowered by casting to float32'


def demo_custom_env_finance_rl_nas89():  # 1.7+ 2.0+
    args = Arguments(if_on_policy=True)  # hyper-parameters of on-policy is different from off-policy
    args.random_seed = 19430

    from FinRLPodracer.elegantrl.agent import AgentPPO
    args.agent = AgentPPO()
    args.agent.lambda_entropy = 0.02

    from FinRLPodracer.finrl.StockTrading import StockEnvNAS89
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
    demo_custom_env_finance_rl_nas89()
