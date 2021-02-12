# Elegant FinRL

  This project is a solution for **FinRL 2.0**: intermediate-level framework for full-stack developers and professionals. 
  
  This project borrows ideas from [ElegantRL](https://github.com/AI4Finance-LLC/ElegantRL) and [FinRL](https://github.com/AI4Finance-LLC/FinRL-Library)
  
  We maintain an **elegant (lightweight, efficient and stable)** FinRL lib, allowing researchers and quant traders to develop algorithimc strategies easily.
  
  + **Lightweight**: The core codes have less than 800 code lines, using PyTorch, OpenAI Gym, and NumPy.
  
  + **Efficient**: Its performance is comparable with Ray RLlib [link](https://github.com/ray-project/ray).
  
  + **Stable**: It is as stable as Stable Baseline 3 [link](https://github.com/DLR-RM/stable-baselines3).
  

# Design Principles

  + **Be Pythonic**: Quant traders, data scientists and machine learning engineers are familiar with the open source Python ecosystem: its programming model, and its tools, e.g., NumPy.
  
  + **Put researchers and algorithmic traders first**: Based on PyTorch, we support researchers to mannually control the execution of the codes, empowering them to improve the performance over automatical libraries.
  
  + **Lean development of algorithmic strategies**: It is better to have an elegant (may be slightly incomplete) solution than a comprehensive but complex and hard to follow design, e.g., Ray RLlib [link](https://github.com/ray-project/ray). It allows fast code iteration.
  
  
# DRL Algorithms

  + **DDPG --> TD3, SAC, A2C, PPO(GAE)**:
  
  + **DQN --> DoubleDQN, DuelingDQN, D3QN**
  
  Please check out OpenAI Spinning Up [DRL Algorithms](https://spinningup.openai.com/en/latest/index.html)


# Training pipeline

+ Initialize the hyper-parameters using `args`.
+ <span style="color:red">初始化 `buffer=ReplayBuffer()` 用于储存回放数据</span>
+ <span style="color:red">Initialize `buffer=ReplayBuffer()` to store the replay buffer.</span>
+ <span style="color:blue">初始化 `agent=AgentXXX()` 用于网络参数更新</span>
+ <span style="color:blue">Initialize `agent=AgentXXX()` to update Neural Network parameters.</span>
+ <span style="color:green">初始化 `recorder=Recorder()` 用于评估并保存模型</span>
+ <span style="color:green">Initialize `recorder=Recorder()` to evaluate and store the model.</span>
+ 训练开始，满足条件后会跳出while循环（条件：达到目标分数、目标步数，或者手动停止）
+ When the training starts, while loop will break after the conditions are met (conditions: achieves the target score, maximum steps, or manually breaks).
  + <span style="color:red">`agent.update_buffer(...)`</span> 智能体在环境中探索规定的步数，产生数据并储存于 `buffer`。可并行
  + <span style="color:red">`agent.update_buffer(...)`</span> The agent explores the environment within target steps, generates data, and stores it in `buffer`. Run in parallel.
  + <span style="color:blue">`agent.update_policy(...)` </span> 智能体使用收集到的数据去更新网络。可并行
  + <span style="color:blue">`agent.update_policy(...)` </span> The agent uses collected data to update the network. Run in parallel.
  + <span style="color:green">recorder.update_recorder(...)</span> 对智能体的性能进行评估，保存分数最高的模型。可独立于训练流程之外
  + <span style="color:green">recorder.update_recorder(...)</span> Evaluate the performance of the agent and keep the model with the highest score. Independent of the training process.


# Code Structure

  + `AgentRun.py`
    + `train_agent()` 配置训练参数，建立训练环境，选择对应算法，然后用这个函数进行训练
    + `train_agent()` sets up hyper-parameters, creates environment, chooses algorithms, and uses the alogrithm to start training.
    + 配置训练参数 `Arguments` 提供了超参数的缺省值与解释。
    + Set up hyper-parameters: `Arguments` provides the default values and corresponding explanations.
    + 建立训练环境 `FinanceMultiStockEnv` 这是一个gym风格的标准训练环境，可以作为模板。    
    + Create environment: `FinanceMultiStockEnv` is a gym-styled standard training environment，and it could be considered as a typical model.    
    + 选择对应算法 `from AgentZoo import AgentXXX` 算法都放在 `AgentZoo.py` 里
    + Choose algorithms: `from AgentZoo import AgentXXX` All available algorithms are inside `AgentZoo.py`.

  + `AgentZoo.py`
    + `AgentBase` 为了适应分布式探索与分布式训练，它需要以下几个功能：
    + `AgentBase` has following functions to adapt distributed exploration and training.
    + 选择探索动作 `select_actions` 离散或连续动作空间、确定或随机策略梯度，他们产生探索动作的方式不同
    + Select exploration actions: `select_actions` has different ways of generating exploration actions for discrete or continuous action space and deterministic or stochastic policy gradient.
    + 更新记忆回放 `update_buffer` off-policy或on-policy 的记忆回放buffer更新机制不同
    + Update replay buffer: `update_buffer` has different updating machenisms for off-policy and on-policy.
    + 加载保存模型 `save_or_load_model` 用来保存分数最高的模型，这样即便后期训练不稳定也没关系
    + Load and save model: `save_or_load_model` is used to save the model with the highest score, which prevents the unstability in the later stage.
    + 创建神经网络 `from AgentNet import XXXnet` 不同DRL算法对应的神经网络放在 `AgentNet.py`
    + Create Neural Network: `from AgentNet import XXXnet` The neural networks corresponding to different DRL algorithms are inside `AgentNet.py`.

  + `AgentNet.py`
    + 所有的基类都继承了 `torch.nn.Module`，这是PyTorch创建神经网络的标准写法
    + All base classes inherit `torch.nn.Module`, which is the standard way of creating neural networks in PyTorch.
    + `QNetBase` 使用`forward()` 输出离散动作的Q值
    + `QNetBase` uses `forward()` to output the Q-values of discrete actions.
    + `ActorBase` 使用`forward()` 输出动作，使用 `get__a_noisy()` 输出带噪声的动作用于探索
    + `ActorBase` uses `forward()` to output actions，and uses `get__a_noisy()` output actions with noises for exploration.
    + `CriticBase` 使用`forward()` 输出Q值的期望
    + `CriticBase` uses `forward()` to output the expected Q-values.


![pipeline](./Readme/pipeline.png)

# API

  + `Env`


