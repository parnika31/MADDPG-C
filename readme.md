# **Multi Agents Constrained Attention Actor Critic**

Code for Constrained MADDPG (MADDPG-C).

Base Code adopted from Shariq Iqbal's [PyTorch implementation](https://github.com/shariqiqbal2810/maddpg-pytorch).

### Requirements 
#
Python 3.7.1 (Minimum) 

PyTorch, version: 1.1.0 (Minimum) 

[OpenAI Baselines](https://github.com/openai/baselines/)

[OpenAI Gym](https://github.com/openai/gym), version: 0.9.4

[Tensorboard](https://github.com/tensorflow/tensorboard), version: 2.1.0

[Tensorboard-PyTorch](https://github.com/lanpa/tensorboardX), version: 1.9

### Training 
#
To train on default hyperparameters(used for the pre-trained models), run the following commands:

For Constrained Cooperative Navigation: `python main.py`

For Constrained Cooperative Treasure Collection: `python main_tc.py`
