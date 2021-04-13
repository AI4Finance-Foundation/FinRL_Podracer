# Elegant FinRL

  This project is a solution for [**FinRL**](https://github.com/AI4Finance-LLC/FinRL-Library) **2.0**: intermediate-level framework for full-stack developers and professionals. 
  
  This project borrows ideas from [ElegantRL](https://github.com/AI4Finance-LLC/ElegantRL) and [FinRL](https://github.com/AI4Finance-LLC/FinRL-Library)
  
  We maintain an **elegant (lightweight, efficient and stable)** FinRL lib, allowing researchers and quant traders to develop algorithmic strategies easily.
  
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
+ <span style="color:red">Initialize `buffer=ReplayBuffer()` : store the transitions.</span>
+ <span style="color:blue">Initialize `agent=AgentXXX()` : update neural network parameters.</span>
+ <span style="color:green">Initialize `recorder=Recorder()` : evaluate and store the trained model.</span>
+ Ater the training starts, the while-loop will break when the conditions are met (conditions: achieving the target score, maximum steps, or manually breaks).
  + <span style="color:red">`agent.update_buffer(...)`</span> The agent explores the environment within target steps, generates transition data, and stores it in the ReplayBuffer. Run in parallel.
  + <span style="color:blue">`agent.update_policy(...)` </span> The agent uses a batch from the ReplayBuffer to update the network parameters. Run in parallel.
  + <span style="color:green">recorder.update_recorder(...)</span> Evaluate the performance of the agent and keep the model with the highest score. Independent of the training process.


# Code Structure

  + `AgentRun.py`
    + `train_agent()` sets up hyper-parameters, creates environment, chooses algorithms, and uses the alogrithm to start training.
    + Set hyper-parameters: `Arguments` provides the default values and corresponding explanations.    
    + Create environment: `FinanceMultiStockEnv` is a gym-styled standard training environment，and it could be considered as a template.    
    + Choose algorithms: `from AgentZoo import AgentXXX` All available algorithms are inside `AgentZoo.py`.

  + `AgentZoo.py`
    + `AgentBase` has following functions to adapt distributed exploration and training.
    + Select exploration actions: `select_actions` has different ways of generating exploration actions for discrete or continuous action spaces, and deterministic or stochastic policy gradients.
    + Update ReplayBuffer: `update_buffer` has different updating machenisms for off-policy and on-policy.
    + Load and save model: `save_or_load_model` is used to save the model with the highest score, which prevents the unstability in the later stage.
    + Create neural wetworks: `from AgentNet import XXXnet` The neural networks corresponding to different DRL algorithms are inside `AgentNet.py`.

  + `AgentNet.py`
    + All base classes inherit `torch.nn.Module`, which is the standard way of creating neural networks in PyTorch.
    + `QNetBase` uses `forward()` to output the Q-values of discrete actions.
    + `ActorBase` uses `forward()` to output actions，and uses `get__a_noisy()` to output actions with noises for exploration.
    + `CriticBase` uses `forward()` to output the expected Q-values.


![pipeline](./Readme/pipeline.png)

# API

  + `Env`


