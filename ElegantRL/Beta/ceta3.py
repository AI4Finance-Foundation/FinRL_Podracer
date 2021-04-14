from elegantrl.run import *


def demo4_bullet_mujoco_off_policy():
    args = Arguments(if_on_policy=False)
    args.random_seed = 100860

    from elegantrl.agent import AgentModSAC  # AgentSAC, AgentTD3, AgentDDPG
    args.agent = AgentModSAC()  # AgentSAC(), AgentTD3(), AgentDDPG()
    args.agent.if_use_dn = True

    import pybullet_envs  # for python-bullet-gym
    dir(pybullet_envs)

    # "TotalStep:  5e4, TargetReturn: 18, UsedTime: 1100s, ReacherBulletEnv-v0"
    # "TotalStep: 30e4, TargetReturn: 25, UsedTime:     s, ReacherBulletEnv-v0"
    # args.env = PreprocessEnv(gym.make('ReacherBulletEnv-v0'))
    # args.env.max_step = 2 ** 10  # important, default env.max_step=150
    # args.reward_scale = 2 ** 0  # -80 < -30 < 18 < 28
    # args.gamma = 0.96
    # args.break_step = int(6e4 * 8)  # (4e4) 8e5, UsedTime: (300s) 700s
    # args.eval_times1 = 2 ** 2
    # args.eval_times1 = 2 ** 5
    # args.if_per = True
    #
    # train_and_evaluate(args)

    "TotalStep:  3e5, TargetReward: 1500, UsedTime:  4ks, AntBulletEnv-v0 ModSAC if_use_dn"
    "TotalStep:  4e5, TargetReward: 2500, UsedTime:  6ks, AntBulletEnv-v0 ModSAC if_use_dn"
    "TotalStep: 15e5, TargetReward: 3198, UsedTime:   ks, AntBulletEnv-v0 ModSAC if_use_dn"
    "TotalStep:  3e5, TargetReward: 1500, UsedTime:  8ks, AntBulletEnv-v0 ModSAC if_use_cn"
    "TotalStep:  7e5, TargetReward: 2500, UsedTime: 18ks, AntBulletEnv-v0 ModSAC if_use_cn"
    "TotalStep: 16e5, TargetReward: 2923, UsedTime:   ks, AntBulletEnv-v0 ModSAC if_use_cn"
    args.env = PreprocessEnv(env=gym.make('AntBulletEnv-v0'))
    args.break_step = int(6e5 * 8)  # (5e5) 1e6, UsedTime: (15,000s) 30,000s
    args.if_allow_break = False
    args.reward_scale = 2 ** -2  # RewardRange: -50 < 0 < 2500 < 3340
    args.max_memo = 2 ** 20
    args.batch_size = 2 ** 9
    args.target_step = args.env.max_step
    args.repeat_times = 2 ** 1
    args.eval_gap = 2 ** 9  # for Recorder
    args.eva_size1 = 2 ** 1  # for Recorder
    args.eva_size2 = 2 ** 3  # for Recorder

    # train_and_evaluate(args)
    args.rollout_num = 4
    train_and_evaluate_mp(args)


if __name__ == '__main__':
    demo4_bullet_mujoco_off_policy()
