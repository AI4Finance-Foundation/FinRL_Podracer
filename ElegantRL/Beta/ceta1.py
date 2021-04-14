from elegantrl.run import *


def demo4_bullet_mujoco_on_policy():
    args = Arguments(if_on_policy=True)  # hyper-parameters of on-policy is different from off-policy

    import pybullet_envs  # for python-bullet-gym
    dir(pybullet_envs)

    # "TotalStep: 1e5, TargetReturn: 18, UsedTime:  3ks, ReacherBulletEnv-v0, PPO"
    # "TotalStep: 1e6, TargetReturn: 18, UsedTime: 30ks, ReacherBulletEnv-v0, PPO"
    # args.env = PreprocessEnv(gym.make('ReacherBulletEnv-v0'))
    #
    # from elegantrl.agent import AgentPPO
    # args.agent = AgentPPO()
    # args.agent.if_use_gae = True
    #
    # args.break_step = int(2e5 * 8)
    # args.reward_scale = 2 ** 0  # RewardRange: -15 < 0 < 18 < 25
    # args.gamma = 0.96
    # args.eval_times1 = 2 ** 2
    # args.eval_times1 = 2 ** 5
    #
    # # train_and_evaluate(args)
    # args.rollout_num = 4
    # train_and_evaluate_mp(args)

    args.env = PreprocessEnv(env=gym.make('HumanoidBulletEnv-v0'))
    args.env.target_return = 2500

    from elegantrl.agent import AgentPPO
    args.agent = AgentPPO()
    args.agent.if_use_gae = True
    args.agent.lambda_entropy = 0.05
    args.agent.lambda_gae_adv = 0.97

    args.if_allow_break = False
    args.break_step = int(8e6 * 8)  # (5e5) 1e6, UsedTime: (15,000s) 30,000s
    args.reward_scale = 2 ** -2  # (-50) 0 ~ 2500 (3340)
    args.max_memo = args.env.max_step * 4
    args.batch_size = 2 ** 11  # 10
    args.repeat_times = 2 ** 3
    args.eval_gap = 2 ** 8  # for Recorder
    args.eva_size1 = 2 ** 1  # for Recorder
    args.eva_size2 = 2 ** 3  # for Recorder

    # train_and_evaluate(args)
    args.rollout_num = 4
    train_and_evaluate_mp(args)


if __name__ == '__main__':
    demo4_bullet_mujoco_on_policy()
