from FinRLPodracer.elegantrl.env import *
from FinRLPodracer.elegantrl.run import *


def get_video_to_watch_gym_render():
    import cv2  # pip3 install opencv-python
    import gym  # pip3 install gym==0.17 pyglet==1.5.0  # env.render() bug in gym==0.18, pyglet==1.6
    import torch

    env_name = 'BipedalWalker-v3'
    net_dim = 2 ** 8
    cwd = f'./{env_name}_4/'
    save_frame_dir = 'frames'

    '''choose env'''
    env = PreprocessEnv(env=gym.make(env_name))
    state_dim = env.state_dim
    action_dim = env.action_dim

    '''initialize agent'''
    from FinRLPodracer.elegantrl.agent import AgentPPO
    agent = AgentPPO()
    agent.if_use_dn = True

    agent.init(net_dim, state_dim, action_dim)
    agent.save_or_load_policy(cwd=cwd, if_save=False)
    device = agent.device

    '''initialize evaluete and env.render()'''

    if save_frame_dir:
        os.makedirs(save_frame_dir, exist_ok=True)

    state = env.reset()
    episode_return = 0
    step = 0
    for i in range(2 ** 9):
        print(i) if i % 128 == 0 else None
        for j in range(1):
            if agent is not None:
                s_tensor = torch.as_tensor((state,), dtype=torch.float32, device=device)
                a_tensor = agent.act(s_tensor)
                action = a_tensor.detach().cpu().numpy()[0]  # if use 'with torch.no_grad()', then '.detach()' not need.
            else:
                action = env.action_space.sample()
            next_state, reward, done, _ = env.step(action)

            episode_return += reward
            step += 1

            if done:
                print(f'{i:>6}, {step:6.0f}, {episode_return:8.3f}, {reward:8.3f}')
                state = env.reset()
                episode_return = 0
                step = 0
            else:
                state = next_state

        frame = env.render('rgb_array')
        frame = frame[50:210, 50:270]  # (240, 320) AntPyBulletEnv-v0
        # frame = cv2.resize(frame[:, :500], (500//2, 720//2))
        cv2.imwrite(f'{save_frame_dir}/{i:06}.png', frame)
        cv2.imshow('', frame)
        cv2.waitKey(1)
    env.close()
    # exit()

    '''convert frames png/jpg to video mp4/avi using ffmpeg'''
    if save_frame_dir:
        frame_shape = cv2.imread(f'{save_frame_dir}/{3:06}.png').shape
        print(f"frame_shape: {frame_shape}")

        save_video = 'gym_render.mp4'
        os.system(f"| Convert frames to video using ffmpeg. Save in {save_video}")
        os.system(f'ffmpeg -r 60 -f image2 -s {frame_shape[0]}x{frame_shape[1]} '
                  f'-i ./{save_frame_dir}/%06d.png '
                  f'-crf 25 -vb 20M -pix_fmt yuv420p {save_video}')


def show_available_env():
    import pybullet_envs  # for python-bullet-gym
    dir(pybullet_envs)

    env_names = list(gym.envs.registry.env_specs.keys())
    env_names.sort()
    for env_name in env_names:
        if env_name.find('Bullet') == -1:
            continue
        print(env_name)


def mp_learner(pipe_net_list=None, learner_id=0):
    print(f'learner_id {learner_id}, {torch.cuda.device_count()}')

    comm = LearnerComm(pipe_net_list, learner_id)
    device = torch.device(f'cuda:{learner_id}' if torch.cuda.is_available() else 'cpu')

    cwd = f'temp/learner_id{learner_id}'
    os.makedirs(cwd, exist_ok=True)
    print(f'learner_id {learner_id}, os.makedirs {cwd}')

    for _ in range(2):
        for round_id in range(comm.round_num):
            data = torch.zeros(1, device=device)
            data = comm.comm_data(data, round_id)

            data = data.to(device)
            print(f'learner_id {learner_id}, {data}')

            time.sleep(0.3 + 0.1 * learner_id)


def check_multiprocessing():
    gpu_id = (0, 1, 2, 3)

    import multiprocessing as mp
    process = list()
    pipe_net_list = [mp.Pipe() for _ in gpu_id]
    for learner_id in range(len(gpu_id)):
        process.append(mp.Process(target=mp_learner, args=(pipe_net_list, learner_id,)))

    [p.start() for p in process]
    [p.join() for p in process]
    [p.terminate() for p in process]


def check_agent():
    from FinRLPodracer.elegantrl.agent import AgentPPO as Agent
    agent = Agent()

    net_dim = 2 ** 7
    state_dim = 8
    action_dim = 2
    agent.init(net_dim, state_dim, action_dim, learning_rate=1e-4, if_use_gae=False, gpu_id=0)

    # print(agent.act.state_dict())
    for key, value in agent.act.state_dict().items():
        print(key, value.shape)


if __name__ == '__main__':
    check_multiprocessing()
