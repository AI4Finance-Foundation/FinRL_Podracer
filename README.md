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
<a href="https://github.com/AI4Finance-LLC/Elegant-FinRL" target="\_blank">
	<div align="center">
		<img src="https://github.com/Yonv1943/ElegantRL/blob/master/figs/File_structure.png" width="100%"/>
	</div>
<!-- 	<div align="center"><caption>Slack Invitation Link</caption></div> -->
</a>
 ![File_structure](https://github.com/Yonv1943/ElegantRL/blob/master/figs/File_structure.png)

   An agent in **agent.py** uses networks in **net.py** and is trained in **run.py** by interacting with an environment in **env.py**.

+ **net.py**    # Neural networks.
   + Q-Net,
   + Actor Network,
   + Critic Network, 
+ **agent.py**  # RL algorithms. 
   + AgentBase 
   + AgentDQN
   + AgentDDPG
   + AgentTD3
   + AgentSAC
   + AgentPPO
+ **env.py** # Stock Trading environment
+ **run.py**    # Stock Trading application
   + Parameter initialization,
   + Training loop,
   + Evaluator.

# Stock Trading Problem Formulation

<a href="https://github.com/AI4Finance-LLC/Elegant-FinRL" target="\_blank">
	<div align="center">
		<img src="figs/1.png" width="40%"/>
	</div>
<!-- 	<div align="center"><caption>Slack Invitation Link</caption></div> -->
</a>

Formally, we model stock trading as a Markov Decision Process (MDP), and formulate the trading objective as maximization of expected return:
+ **State s = [b, p, h]**: a vector that includes the remaining balance b, stock prices p, and stock shares h. p and h are vectors with D dimension, where D denotes the number of stocks.
+ **Action a**: a vector of actions over D stocks. The allowed actions on each stock include selling, buying, or holding, which result in decreasing, increasing, or no change of the stock shares in h, respectively.
+ **Reward r(s, a, s’)**: The asset value change of taking action a at state s and arriving at new state s’.
+ **Policy π(s)**: The trading strategy at state s, which is a probability distribution of actions at state s.
+ **Q-function Q(s, a)**: the expected return (reward) of taking action a at state s following policy π.
+ **State-transition**: After taking the actions a, the number of shares h is modified, as shown in Fig 3, and the new portfolio is the summation of the balance and the total value of the stocks.

<a href="https://github.com/AI4Finance-LLC/Elegant-FinRL" target="\_blank">
	<div align="center">
		<img src="figs/2.png" width="50%"/>
	</div>
<!-- 	<div align="center"><caption>Slack Invitation Link</caption></div> -->
</a>

