import os
import time
import torch
import numpy as np


class Evaluator:
    def __init__(self, cwd, agent_id, eval_times1, eval_times2, eval_gap, env, device):
        self.recorder = list()  # total_step, r_avg, r_std, obj_c, ...
        self.recorder_path = f'{cwd}/recorder.npy'
        self.r_max = -np.inf
        self.total_step = 0

        self.env = env
        self.cwd = cwd
        self.device = device
        self.agent_id = agent_id
        self.eval_gap = eval_gap
        self.eval_times1 = eval_times1
        self.eval_times2 = eval_times2
        self.target_return = env.target_return

        self.used_time = None
        self.start_time = time.time()
        self.eval_time = 0
        print(f"{'#' * 80}\n"
              f"{'ID':<3}{'Step':>8}{'maxR':>8} |"
              f"{'avgR':>8}{'stdR':>7}{'avgS':>7}{'stdS':>6} |"
              f"{'expR':>8}{'objC':>7}{'etc.':>7}")

    def save_or_load_recoder(self, if_save):
        if if_save:
            np.save(self.recorder_path, self.recorder)
        elif os.path.exists(self.recorder_path):
            recorder = np.load(self.recorder_path)
            self.recorder = [tuple(i) for i in recorder]  # convert numpy to list
            self.total_step = self.recorder[-1][0]

    def evaluate_and_save(self, act, steps, r_exp, log_tuple) -> bool:
        self.total_step += steps  # update total training steps

        if time.time() - self.eval_time < self.eval_gap:
            return False  # if_reach_goal

        self.eval_time = time.time()
        rewards_steps_list = [get_episode_return_and_step(self.env, act, self.device) for _ in
                              range(self.eval_times1)]
        r_avg, r_std, s_avg, s_std = self.get_r_avg_std_s_avg_std(rewards_steps_list)

        if r_avg > self.r_max:  # evaluate actor twice to save CPU Usage and keep precision
            rewards_steps_list += [get_episode_return_and_step(self.env, act, self.device)
                                   for _ in range(self.eval_times2 - self.eval_times1)]
            r_avg, r_std, s_avg, s_std = self.get_r_avg_std_s_avg_std(rewards_steps_list)
        if r_avg > self.r_max:  # save checkpoint with highest episode return
            self.r_max = r_avg  # update max reward (episode return)

            act_save_path = f'{self.cwd}/actor.pth'
            torch.save(act.state_dict(), act_save_path)  # save policy network in *.pth
            print(f"{self.agent_id:<3}{self.total_step:8.2e}{self.r_max:8.2f} |")  # save policy and print

        self.recorder.append((self.total_step, r_avg, r_std, *log_tuple))  # update recorder

        if_reach_goal = bool(self.r_max > self.target_return)  # check if_reach_goal
        if if_reach_goal and self.used_time is None:
            self.used_time = int(time.time() - self.start_time)
            print(f"{'ID':<3}{'Step':>8}{'TargetR':>8} |"
                  f"{'avgR':>8}{'stdR':>7}{'avgS':>7}{'stdS':>6} |"
                  f"{'UsedTime':>8}  ########\n"
                  f"{self.agent_id:<3}{self.total_step:8.2e}{self.target_return:8.2f} |"
                  f"{r_avg:8.2f}{r_std:7.1f}{s_avg:7.0f}{s_std:6.0f} |"
                  f"{self.used_time:>8}  ########")

        print(f"{self.agent_id:<3}{self.total_step:8.2e}{self.r_max:8.2f} |"
              f"{r_avg:8.2f}{r_std:7.1f}{s_avg:7.0f}{s_std:6.0f} |"
              f"{r_exp:8.2f}{''.join(f'{n:7.2f}' for n in log_tuple)}")
        self.draw_plot()
        return if_reach_goal

    def draw_plot(self):
        if len(self.recorder) == 0:
            print("| save_npy_draw_plot() WARNNING: len(self.recorder)==0")
            return None

        np.save(self.recorder_path, self.recorder)

        '''draw plot and save as png'''
        train_time = int(time.time() - self.start_time)
        total_step = int(self.recorder[-1][0])
        save_title = f"step_time_maxR_{int(total_step)}_{int(train_time)}_{self.r_max:.3f}"

        save_learning_curve(self.recorder, self.cwd, save_title)

    @staticmethod
    def get_r_avg_std_s_avg_std(rewards_steps_list):
        rewards_steps_ary = np.array(rewards_steps_list, dtype=np.float)
        r_avg, s_avg = rewards_steps_ary.mean(axis=0)  # average of episode return and episode step
        r_std, s_std = rewards_steps_ary.std(axis=0)  # standard dev. of episode return and episode step
        return r_avg, r_std, s_avg, s_std


def get_episode_return_and_step(env, act, device) -> (float, int):
    episode_return = 0.0  # sum of rewards in an episode
    episode_step = 1
    max_step = env.max_step
    if_discrete = env.if_discrete

    state = env.reset()
    for episode_step in range(max_step):
        s_tensor = torch.as_tensor((state,), device=device)
        a_tensor = act(s_tensor)
        if if_discrete:
            a_tensor = a_tensor.argmax(dim=1)
        action = a_tensor.detach().cpu().numpy()[0]  # not need detach(), because with torch.no_grad() outside
        state, reward, done, _ = env.step(action)
        episode_return += reward
        if done:
            break
    episode_return = getattr(env, 'episode_return', episode_return)
    return episode_return, episode_step


def save_learning_curve(recorder, cwd='.', save_title='learning curve', fig_name='plot_learning_curve.jpg'):
    recorder = np.array(recorder)  # recorder_ary.append((self.total_step, r_avg, r_std, obj_a, obj_c))
    steps = recorder[:, 0]  # x-axis is training steps
    r_avg = recorder[:, 1]
    r_std = recorder[:, 2]
    obj_c = recorder[:, 3]
    obj_a = recorder[:, 4]

    '''plot subplots'''
    import matplotlib as mpl
    mpl.use('Agg')
    """Generating matplotlib graphs without a running X server [duplicate]
    write `mpl.use('Agg')` before `import matplotlib.pyplot as plt`
    https://stackoverflow.com/a/4935945/9293137
    """
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2)

    axs0 = axs[0]
    axs0.cla()
    color0 = 'lightcoral'
    axs0.set_xlabel('Total Steps')
    axs0.set_ylabel('Episode Return')
    axs0.plot(steps, r_avg, label='Episode Return', color=color0)
    axs0.fill_between(steps, r_avg - r_std, r_avg + r_std, facecolor=color0, alpha=0.3)

    ax11 = axs[1]
    ax11.cla()
    color11 = 'royalblue'
    axs0.set_xlabel('Total Steps')
    ax11.set_ylabel('objA', color=color11)
    ax11.plot(steps, obj_a, label='objA', color=color11)
    ax11.tick_params(axis='y', labelcolor=color11)
    for plot_i in range(5, recorder.shape[1]):
        other = recorder[:, plot_i]
        ax11.plot(steps, other, label=f'{plot_i}', color='grey')

    ax12 = axs[1].twinx()
    color12 = 'darkcyan'
    ax12.set_ylabel('objC', color=color12)
    ax12.fill_between(steps, obj_c, facecolor=color12, alpha=0.2, )
    ax12.tick_params(axis='y', labelcolor=color12)

    '''plot save'''
    plt.title(save_title, y=2.3)
    plt.savefig(f"{cwd}/{fig_name}")
    plt.close('all')  # avoiding warning about too many open figures, rcParam `figure.max_open_warning`
    # plt.show()  # if use `mpl.use('Agg')` to draw figures without GUI, then plt can't plt.show()
