import time

import gym as gym
import numpy as np
import torch
import torch.nn as nn
import torch.distributions as td
from gym.wrappers import GrayScaleObservation, FlattenObservation, TransformObservation, ResizeObservation
from gym.wrappers.pixel_observation import PixelObservationWrapper

from model import DVBF
from wrapper import PixelDictWrapper, PendulumEnv

dim_z = 3
dim_x = (16, 16)
dim_u = 1
dim_a = 16
dim_w = 3
batch_size = 500
num_iterations = int(10**5)
learning_rate = 0.1



q_params = nn.Sequential(
    nn.Linear(in_features=dim_z+np.prod(dim_x).item()+dim_u, out_features=128),
    nn.ReLU(),
    nn.Linear(in_features=128, out_features=2*dim_w)
)

transition_params = nn.Sequential(
    nn.Linear(in_features=dim_z, out_features=dim_a),
    nn.Softmax()
)

initial_w = nn.LSTM(input_size=np.prod(dim_x).item(), batch_first=True, hidden_size=128, dropout=0.1, proj_size=3, bidirectional=True)
initial_transition = nn.Sequential(
    nn.Linear(in_features=dim_w, out_features=128),
    nn.ReLU(),
    nn.Linear(in_features=128, out_features=dim_z)
)


def make_env():
    env = PendulumEnv()
    env.reset()
    env = PixelDictWrapper(PixelObservationWrapper(env))
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=(16, 16))
    env = FlattenObservation(env)
    print(env.action_space)
    print(env.observation_space)
    return env

def collect_data(num_sequences: int, sequence_length: int):
    data = []
    env = make_env()
    while len(data) < num_sequences:
        obs = env.reset()
        done, t = False, 0
        episode = []
        while not done and len(data) < num_sequences:
            action = env.action_space.sample()
            episode.append(np.concatenate([obs, action]))
            obs, reward, done, info = env.step(action)
            t += 1
            if t == sequence_length:
                t = 0
                data.append(episode)
                episode = []
    np.savez(f'dataset/validation.npz', data=np.array(data))

def load_data(file: str) -> torch.Tensor:
    return torch.from_numpy(np.load(file)['data'])

if __name__ == '__main__':
    train, test, validation = load_data('dataset/training.npz'), load_data('dataset/test.npz'), load_data(
        'dataset/validation.npz')

    x = train[..., :-1]
    u = train[..., -1:]
    dvbf = DVBF(dim_x=x.shape[-1], dim_u=1, dim_z=4, dim_w=4)

    optimizer = torch.optim.Adadelta(dvbf.parameters(), lr=0.01)
    for i in range(num_iterations):
        optimizer.zero_grad()
        loss = dvbf.fit(x, u)
        loss.backward()
        optimizer.step()

    x = 0