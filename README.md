# Elegant FinRL

  This project is a solution for [**FinRL**](https://github.com/AI4Finance-LLC/FinRL-Library) **2.0**: intermediate-level framework for full-stack developers and professionals. 
  
  This project borrows ideas from [ElegantRL](https://github.com/AI4Finance-LLC/ElegantRL) and [FinRL](https://github.com/AI4Finance-LLC/FinRL-Library)
  
  We maintain an **elegant (lightweight, efficient and stable)** FinRL lib, allowing researchers and quant traders to develop algorithmic strategies easily.
  
  + **Lightweight**: The core codes have less than 800 code lines, using PyTorch and NumPy.
  
  + **Efficient**: Its performance is comparable with [Ray RLlib](https://github.com/ray-project/ray).
  
  + **Stable**: It is as stable as [Stable Baseline 3](https://github.com/DLR-RM/stable-baselines3).
  

# Design Principles

  + **Be Pythonic**: Quant traders, data scientists and machine learning engineers are familiar with the open source Python ecosystem: its programming model, and its tools, e.g., NumPy.
  
  + **Put researchers and algorithmic traders first**: Based on PyTorch, we support researchers to mannually control the execution of the codes, empowering them to improve the performance over automatical libraries.
  
  + **Lean development of algorithmic strategies**: It is better to have an elegant (may be slightly incomplete) solution than a comprehensive but complex and hard to follow design, e.g., Ray RLlib [link](https://github.com/ray-project/ray). It allows fast code iteration.
  
  
# DRL Algorithms

Currently, model-free deep reinforcement learning (DRL) algorithms: 
+ **DDPG, TD3, SAC, A2C, PPO, PPO(GAE)** for continuous actions
+ **DQN, DoubleDQN, D3QN** for discrete actions

For DRL algorithms, please check out the educational webpage [OpenAI Spinning Up](https://spinningup.openai.com/en/latest/). 

# File Structure
 ![File_structure](https://github.com/Yonv1943/ElegantRL/blob/master/figs/File_structure.png)

   An agent in **agent.py** uses networks in **net.py** and is trained in **run.py** by interacting with an environment in **env.py**.

+ **net.py**    # Neural networks.
   + Q-Net,
   + Actor Network,
   + Critic Network, 
+ **agent.py**  # RL algorithms. 
   + AgentBase 
+ **env.py** # Stock Trading environment
+ **run.py**    # Stock Trading application
   + Parameter initialization,
   + Training loop,
   + Evaluator.


