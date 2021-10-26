import os

import torch
import numpy as np
from copy import deepcopy
from FinRLPodracer.elegantrl.net import ActorPPO, CriticPPO


class AgentBase:
    """
    Base Class
    """
    def __init__(self):
        self.state = None
        self.device = None
        self.action_dim = None
        self.if_on_policy = None
        self.get_obj_critic = None

        self.criterion = torch.nn.SmoothL1Loss()
        self.cri = self.cri_optim = self.Cri = None  # self.Cri is the class of cri
        self.act = self.act_optim = self.Act = None  # self.Act is the class of cri
        self.cri_target = self.if_use_cri_target = None
        self.act_target = self.if_use_act_target = None

    def init(self, net_dim, state_dim, action_dim, learning_rate=1e-4, if_use_per=False, gpu_id=0):
        """
        Explict call ``self.init()`` to overwrite the ``self.object`` in ``__init__()`` for multiprocessing.
        """
        self.device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu")
        self.action_dim = action_dim

        self.cri = self.Cri(net_dim, state_dim, action_dim).to(self.device)
        self.act = self.Act(net_dim, state_dim, action_dim).to(self.device) if self.Act is not None else self.cri
        self.cri_target = deepcopy(self.cri) if self.if_use_cri_target else self.cri
        self.act_target = deepcopy(self.act) if self.if_use_act_target else self.act

        self.cri_optim = torch.optim.Adam(self.cri.parameters(), learning_rate)
        self.act_optim = torch.optim.Adam(self.act.parameters(), learning_rate) if self.Act is not None else self.cri

        del self.Cri, self.Act

    def select_action(self, state) -> np.ndarray:
        """
        Select actions given a state.

        :param state: a state in a shape (state_dim, ).
        :return: an actions in a shape (action_dim, ) where each action is clipped into range(-1, 1).
        """
        pass  # sample form an action distribution

    def explore_env(self, env, target_step, reward_scale, gamma) -> list:
        """
        Collect trajectories through the actor-environment interaction for a **single** environment instance.

        :param env: the DRL environment instance.
        :param target_step: the total step for the interaction.
        :param reward_scale: a reward scalar to clip the reward.
        :param gamma: the discount factor.
        :return: a list of trajectories [traj, ...] where each trajectory is a list of transitions [(state, other), ...].
        """
        trajectory_list = list()

        state = self.state
        for _ in range(target_step):
            action = self.select_action(state)
            next_s, reward, done, _ = env.step(action)
            other = (reward * reward_scale, 0.0 if done else gamma, *action)
            trajectory_list.append((state, other))

            state = env.reset() if done else next_s
        self.state = state
        return trajectory_list

    def update_net(self, buffer, batch_size, repeat_times, soft_update_tau) -> tuple:
        """
        Update the neural networks by sampling batch data from ``ReplayBuffer``.

        :param buffer: the ReplayBuffer instance that stores the trajectories.
        :param batch_size: the size of batch data for Stochastic Gradient Descent (SGD).
        :param repeat_times: the re-using times of each trajectory.
        :param soft_update_tau: the soft update parameter.
        :return: a tuple of the log information.
        """

    @staticmethod
    def optim_update(optimizer, objective):
        optimizer.zero_grad()
        objective.backward()
        optimizer.step()

    @staticmethod
    def optim_update_amp(optimizer, objective):  # automatic mixed precision
        pass

    @staticmethod
    def soft_update(target_net, current_net, tau):
        for tar, cur in zip(target_net.parameters(), current_net.parameters()):
            tar.data.copy_(cur.data * tau + tar.data * (1 - tau))

    def save_or_load_agent(self, cwd, if_save):
        def load_torch_file(model_or_optim, _path):
            state_dict = torch.load(_path, map_location=lambda storage, loc: storage)
            model_or_optim.load_state_dict(state_dict)

        name_obj_list = [('actor', self.act),
                         ('critic', self.cri),
                         ('act_target', self.act_target),
                         ('cri_target', self.cri_target),
                         ('act_optim', self.act_optim),
                         ('cri_optim', self.cri_optim), ]
        name_obj_list = [(name, obj)
                         for name, obj in name_obj_list
                         if obj is not None]

        if if_save:
            for name, obj in name_obj_list:
                save_path = f"{cwd}/{name}.pth"
                torch.save(obj.state_dict(), save_path)
            print(f"| Agent save: {cwd}")
        else:
            for name, obj in name_obj_list:
                save_path = f"{cwd}/{name}.pth"
                load_torch_file(obj, save_path) if os.path.isfile(save_path) else None
            # print(f"| Agent load: {cwd}")


class AgentPPO(AgentBase):
    def __init__(self):
        super().__init__()
        self.if_on_policy = True

        self.ratio_clip = 0.25  # ratio.clamp(1 - clip, 1 + clip)
        self.lambda_entropy = 0.02  # could be 0.02
        self.lambda_gae_adv = 0.98  # could be 0.95~0.99, GAE (Generalized Advantage Estimation. ICLR.2016.)
        self.get_reward_sum = None
        self.srdan_list = list()

        self.Act = ActorPPO
        self.Cri = CriticPPO

    def init(self, net_dim, state_dim, action_dim, learning_rate=1e-4, if_use_gae=False, gpu_id=0):
        super().init(net_dim, state_dim, action_dim, learning_rate, if_use_gae, gpu_id)
        self.get_reward_sum = self.get_reward_sum_gae if if_use_gae else self.get_reward_sum_raw

    def select_action(self, state):
        states = torch.as_tensor((state,), dtype=torch.float32, device=self.device)
        actions, noises = self.act.get_action(states)  # plan to be get_action_a_noise
        return actions[0].detach().cpu().numpy(), noises[0].detach().cpu().numpy()

    def select_actions(self, states):
        actions, noises = self.act.get_action(states)  # plan to be get_action_a_noise
        return actions, noises

    def explore_env(self, env, target_step, reward_scale, gamma):
        state = self.state
        srdan_list = list()

        last_done = 0
        for i in range(target_step):
            action, noise = self.select_action(state)
            next_state, reward, done, _ = env.step(np.tanh(action))
            srdan_list.append((state, reward, done, action, noise))

            if done:
                state = env.reset()
                last_done = i
            else:
                state = next_state
        self.state = state

        last_done += 1
        srdan_list, self.srdan_list = self.srdan_list + srdan_list[:last_done], srdan_list[last_done:]

        srdan_list = list(map(list, zip(*srdan_list)))  # 2D-list transpose
        srdan_list = [torch.as_tensor(t, dtype=torch.float32, device=self.device)
                      for t in srdan_list]
        states, rewards, dones, actions, noises = srdan_list
        rewards *= reward_scale
        masks = (1 - dones) * gamma
        return states, rewards, masks, actions, noises

    def explore_envs_check(self, env, target_step, reward_scale, gamma):
        print(';', 0, target_step)

        state = self.state
        env_num = env.env_num

        states = torch.empty((target_step, env_num, env.state_dim), dtype=torch.float32, device=self.device)
        actions = torch.empty((target_step, env_num, env.action_dim), dtype=torch.float32, device=self.device)
        noises = torch.empty((target_step, env_num, env.action_dim), dtype=torch.float32, device=self.device)
        rewards = torch.empty((target_step, env_num), dtype=torch.float32, device=self.device)
        dones = torch.empty((target_step, env_num), dtype=torch.float32, device=self.device)
        for i in range(target_step):
            action, noise = self.select_actions(state)
            states[i] = state  # previous state

            state, reward, done, _ = env.step_vec(action.tanh())  # next_state
            actions[i] = action
            noises[i] = noise
            rewards[i] = reward
            dones[i] = done
        self.state = state

        rewards *= reward_scale
        masks = (1 - dones) * gamma
        return states, rewards, masks, actions, noises

    def prepare_buffer(self, s_r_m_a_n_list):
        with torch.no_grad():  # compute reverse reward
            state, reward, mask, action, noise = s_r_m_a_n_list
            buf_len = state.size(0)

            bs = 2 ** 10  # set a smaller 'BatchSize' when out of GPU memory.
            value = torch.cat([self.cri_target(state[i:i + bs]) for i in range(0, state.size(0), bs)], dim=0)
            logprob = self.act.get_old_logprob(action, noise)

            r_sum, advantage = self.get_reward_sum(buf_len, reward, mask, value)
        return state, action, r_sum, logprob, advantage

    def update_net(self, buffer, batch_size, repeat_times, soft_update_tau):
        with torch.no_grad():
            if isinstance(buffer[0], tuple):
                buffer = list(map(list, zip(*buffer)))  # 2D-list transpose
                buffer = [torch.cat(tensor_list, dim=0).to(self.device)
                          for tensor_list in buffer]
            buf_state, buf_action, buf_r_sum, buf_logprob, buf_advantage = buffer
            buf_advantage = (buf_advantage - buf_advantage.mean()) / (buf_advantage.std() + 1e-5)
        buf_len = buf_state.size(0)

        '''PPO: Surrogate objective of Trust Region'''
        div_r_sum_std = (1 / buf_r_sum.std() + 1e-6)
        obj_critic = obj_actor = old_logprob = None
        for _ in range(int(buf_len / batch_size * repeat_times)):
            indices = torch.randint(buf_len, size=(batch_size,), requires_grad=False, device=self.device)

            state = buf_state[indices]
            action = buf_action[indices]
            r_sum = buf_r_sum[indices]
            old_logprob = buf_logprob[indices]
            advantage = buf_advantage[indices]

            new_logprob, obj_entropy = self.act.get_logprob_entropy(state, action)  # it is obj_actor
            ratio = (new_logprob - old_logprob.detach()).exp()
            surrogate1 = advantage * ratio
            surrogate2 = advantage * ratio.clamp(1 - self.ratio_clip, 1 + self.ratio_clip)
            obj_surrogate = -torch.min(surrogate1, surrogate2).mean()
            obj_actor = obj_surrogate + obj_entropy * self.lambda_entropy
            self.optim_update(self.act_optim, obj_actor)

            value = self.cri(state).squeeze(1)  # critic network predicts the reward_sum (Q value) of state
            obj_critic = self.criterion(value, r_sum)
            self.optim_update(self.cri_optim, obj_critic * div_r_sum_std)
            self.soft_update(self.cri_target, self.cri, soft_update_tau) if self.cri_target is not self.cri else None

        return obj_critic.item(), obj_actor.item(), old_logprob.mean().item() / self.action_dim  # logging_tuple

    def get_reward_sum_raw(self, buf_len, buf_reward, buf_mask, buf_value) -> (torch.Tensor, torch.Tensor):
        buf_r_sum = torch.empty(buf_len, dtype=torch.float32, device=self.device)  # reward sum

        pre_r_sum = 0
        for i in range(buf_len - 1, -1, -1):
            buf_r_sum[i] = buf_reward[i] + buf_mask[i] * pre_r_sum
            pre_r_sum = buf_r_sum[i]
        buf_advantage = buf_r_sum - (buf_mask * buf_value.squeeze(1))
        return buf_r_sum, buf_advantage

    def get_reward_sum_gae(self, buf_len, buf_reward, buf_mask, buf_value) -> (torch.Tensor, torch.Tensor):
        buf_r_sum = torch.empty(buf_len, dtype=torch.float32, device=self.device)  # old policy value
        buf_advantage = torch.empty(buf_len, dtype=torch.float32, device=self.device)  # advantage value

        pre_r_sum = 0
        pre_advantage = 0  # advantage value of previous step
        for i in range(buf_len - 1, -1, -1):
            buf_r_sum[i] = buf_reward[i] + buf_mask[i] * pre_r_sum
            pre_r_sum = buf_r_sum[i]

            buf_advantage[i] = buf_reward[i] + buf_mask[i] * (pre_advantage - buf_value[i])  # fix a bug here
            pre_advantage = buf_value[i] + buf_advantage[i] * self.lambda_gae_adv
        return buf_r_sum, buf_advantage
