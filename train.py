import time

import cv2
import gym as gym
import imageio
import numpy as np
import torch
import torch.nn as nn
import torch.distributions as td
from torch.optim.lr_scheduler import ExponentialLR
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader, Dataset
from gym.wrappers import GrayScaleObservation, FlattenObservation, TransformObservation, ResizeObservation
from gym.wrappers.pixel_observation import PixelObservationWrapper

from model import DVBF
from wrapper import PixelDictWrapper, PendulumEnv

dim_z = 3
dim_x = (16, 16)
dim_u = 1
dim_a = 16
dim_w = 3
batch_size = 32
num_iterations = int(5000)
learning_rate = 0.01


def make_env():
    env = PendulumEnv()
    env.reset()
    env = PixelDictWrapper(PixelObservationWrapper(env))
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=(16, 16))

    print(env.action_space)
    print(env.observation_space)
    return env

def collect_data(num_sequences: int, sequence_length: int):
    data = dict(obs=[], actions=[])
    env = make_env()
    episodes = 0
    while episodes < num_sequences:
        obs = env.reset()
        done, t = False, 0
        observations, actions = [], []
        while not done and episodes < num_sequences:
            action = env.action_space.sample()
            observations.append(obs)
            actions.append(action)
            obs, reward, done, info = env.step(action)
            t += 1
            if t == sequence_length:
                t = 0
                data['obs'].append(observations)
                data['actions'].append(actions)
                observations, actions = [], []
                episodes += 1
    np.savez(f'dataset/raw.npz', obs=np.array(data['obs']), actions=np.asarray(data['actions']))

def load_data(file: str, device='cpu') -> Dataset:
    data = torch.from_numpy(np.load(file)['data']).to(device)
    x, u = data[..., :-1], data[..., -1:]
    x = x.to(torch.float32) / 255. - 0.5
    return TensorDataset(x, u)

def train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    writer = SummaryWriter()
    datasets = dict((k, load_data(file=f'dataset/{k}.npz', device=device)) for k in ['training', 'test', 'validation'])

    train_loader = DataLoader(datasets['training'], batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(datasets['validation'], batch_size=batch_size, shuffle=False)

    dvbf = DVBF(dim_x=16*16, dim_u=1, dim_z=4, dim_w=4).to(device)
    #dvbf = torch.load('dvbf.th').to(device)

    optimizer = torch.optim.Adam(dvbf.parameters(), lr=learning_rate)
    scheduler = ExponentialLR(optimizer, gamma=0.9)
    for i in range(num_iterations):
        total_loss = 0

        dvbf.train()
        for batch in train_loader:
            x, u = batch[0], batch[1]
            optimizer.zero_grad()
            loss = dvbf.loss(x, u)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        writer.add_scalar('loss', scalar_value=total_loss, global_step=i)
        writer.add_scalar('learning rate', scalar_value=scheduler.get_lr()[0], global_step=i)

        dvbf.train(False)
        total_val_loss = 0
        for batch in validation_loader:
            x, u = batch[0], batch[1]
            val_loss = dvbf.loss(x, u)
            total_val_loss += val_loss.item()
        writer.add_scalar('val_loss', scalar_value=total_val_loss, global_step=i)
        print(f'[Epoch {i}] train_loss: {total_loss}, val_loss: {total_val_loss}')

        if i % 100 == 0:
            torch.save(dvbf, 'checkpoints/dvbf.th')
            generate(filename=f'dvbf-epoch-{i}')

    torch.save(dvbf, 'dvbf.th')

def generate(filename):
    dvbf = torch.load('dvbf.th').to('cpu')
    dataset = load_data('dataset/validation.npz')
    x = dataset[0][0].unsqueeze(dim=0)
    u = dataset[0][1].unsqueeze(dim=0)
    T = u.shape[1]
    z, _ = dvbf.filter(x=x[:1, :5], u=u)
    reconstructed = dvbf.reconstruct(z).view(1, T, -1)

    def format(x):
        img = torch.clip((x + 0.5) * 255., 0, 255).to(torch.uint8)
        return img.view(-1, 16, 16).numpy()
    frames = []
    for i in range(T):
        gt = format(x[:, i])
        pred = format(reconstructed[:, i])
        img = np.concatenate([gt, pred], axis=1).squeeze()
        #cv2.imshow(mat=img, winname='generated')
        #cv2.waitKey(50)
        frames.append(img)
    with imageio.get_writer(f"checkpoints/{filename}.mp4", mode="I") as writer:
        for idx, frame in enumerate(frames):
            writer.append_data(frame)


if __name__ == '__main__':
    #collect_data(5, 15)
    train()
    generate()