from elegantrl.run import *


def demo4_bullet_mujoco_on_policy():
    args = Arguments(if_on_policy=True)  # hyper-parameters of on-policy is different from off-policy
    args.random_seed = 104367

    import pybullet_envs  # for python-bullet-gym
    dir(pybullet_envs)

    "TotalStep:  6e6, TargetReturn: 1500, UsedTime:  5ks, HumanoidBulletEnv-v0, PPO"
    "TotalStep: 12e6, TargetReturn: 2500, UsedTime: 10ks, HumanoidBulletEnv-v0, PPO"
    "TotalStep: 51e6, TargetReturn: 3077, UsedTime: 40ks, HumanoidBulletEnv-v0, PPO"
    args.env = PreprocessEnv(env=gym.make('HumanoidBulletEnv-v0'))
    args.env.target_return = 2500

    from elegantrl.agent import AgentPPO
    args.agent = AgentPPO()
    args.agent.if_use_gae = True
    args.agent.lambda_entropy = 0.05
    args.agent.lambda_gae_adv = 0.97

    args.if_allow_break = False
    args.break_step = int(8e6 * 8)  # (5e5) 1e6, UsedTime: (15,000s) 30,000s
    args.reward_scale = 2 ** -3  # (-50) 0 ~ 2500 (3340)
    args.max_memo = args.env.max_step * 4
    args.batch_size = 2 ** 11  # 10
    args.repeat_times = 2 ** 3
    args.eval_gap = 2 ** 9  # for Recorder
    args.eva_size1 = 2 ** 1  # for Recorder
    args.eva_size2 = 2 ** 3  # for Recorder

    # train_and_evaluate(args)
    args.rollout_num = 4
    train_and_evaluate_mp(args)


if __name__ == '__main__':
    demo4_bullet_mujoco_on_policy()
