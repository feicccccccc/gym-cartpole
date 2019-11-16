"""

testing random policy search on cart-pole problem:

https://gym.openai.com/envs/CartPole-v0/

Gym env source code:
https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py#L75

credit:
https://medium.com/@m.alzantot/deep-reinforcement-learning-demystified-episode-0-2198c05a6124

"""

import gym
import numpy as np


env = gym.make('CartPole-v0')
observation = env.reset()

"""
Input Space dimension
"""

# print(env.action_space) # Discrete(2)
# Int 0 -> go left
# Int 1 -> go right

"""
Output Space dimension
from env source code:

Observation: 
        Type: Box(4)
        Num	Observation                 Min         Max
        0	Cart Position             -4.8            4.8
        1	Cart Velocity             -Inf            Inf
        2	Pole Angle                 -24 deg        24 deg
        3	Pole Velocity At Tip      -Inf            Inf
        
"""

# print(env.observation_space) # Box(4,)

"""
Reward: +1 for every time step
"""

"""
Hyperparameter:

"""

num_polciy = 500 # numnber of polciy to search
render = True # Render or not
max_iteration = 1000

"""
Model:

linear combination of the 4 observation + bais
i.e.

w * x + b > 0 -> output 1
w * x + b < 0 -> output 0

weight = (4,)
bias = float

# Sounds good to use gradient descent to find the parameter

"""

def gen_random_policy():
    # (w,b)
	return (np.random.uniform(-1,1, size=4), np.random.uniform(-1,1))

def run_episode(env, policy, t_max = max_iteration, render=False):
    # t_max : max time step
    obs = env.reset()
    total_reward = 0
    for i in range(t_max):
        if render:
            env.render()
        selected_action = 1 if np.dot(policy[0], obs) + policy[1] > 0 else 0
        # action to take from base on weight and bias
        obs, reward, done, _ = env.step(selected_action)
        total_reward += reward
        if done:
            print("Reward = {}".format(total_reward))
            break
    return total_reward

# generate list of random weight and bias
policy_list = [gen_random_policy() for _ in range(num_polciy)]

# generate list of random result using those weight and bias
scores_list = [run_episode(env, p) for p in policy_list]

print("Best para in the pool is {}".format(scores_list.index(max(scores_list))))

print('Running with best policy:\n')
run_episode(env, policy_list[scores_list.index(max(scores_list))],t_max=10000000,render=True)

env.close()