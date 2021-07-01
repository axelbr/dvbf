from typing import Tuple, Dict

import torch
import torch.nn as nn
import numpy as np


class DVBF(nn.Module):
    def __init__(self, dim_x: Tuple, dim_u: int, dim_z: int, dim_w: int, num_matrices: int = 16, hidden_size=128):
        super().__init__()
        self.dim_x = np.prod(dim_x).item()
        self.dim_z = dim_z
        self.dim_w = dim_w
        self.dim_u = dim_u
        self.initial_lstm = nn.LSTM(input_size=dim_x, batch_first=True, hidden_size=hidden_size, dropout=0.1, bidirectional=True)
        self.initial_to_params = nn.Sequential(
            nn.Linear(in_features=2*hidden_size, out_features=hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size, out_features=2*dim_w)
        )
        self.w1_to_z1 = nn.Sequential(
            nn.Linear(in_features=dim_w, out_features=hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size, out_features=dim_z)
        )
        self.w_params = nn.Sequential(
            nn.Linear(in_features=dim_z+dim_u+self.dim_x, out_features=hidden_size),
            nn.Tanh(),
            nn.Linear(in_features=hidden_size, out_features=2*dim_w),
        )

        self.v_params = nn.Sequential(
            nn.Linear(in_features=dim_z+dim_u, out_features=hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size, out_features=num_matrices),
            nn.Softmax()
        )

        self.observation_model = nn.Sequential(
            nn.Linear(in_features=dim_z, out_features=hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size, out_features=self.dim_x)
        )

        self.A = torch.randn([num_matrices, dim_z, dim_z], requires_grad=True)
        self.B = torch.randn([num_matrices, dim_z, dim_u], requires_grad=True)
        self.C = torch.randn([num_matrices, dim_z, dim_w], requires_grad=True)


    def get_initial_samples(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        output, (hidden, cell_states) = self.initial_lstm(x)
        w_params = self.initial_to_params(output[:, -1])
        mean, std = torch.split(w_params, split_size_or_sections=self.dim_w, dim=1)
        std = torch.exp(std) + 1e-5
        w1 = torch.normal(mean, std)
        z1 = self.w1_to_z1(w1)
        return w1, z1, dict(mean=mean, std=std)

    def mix_matrices(self, z_t, u_t):
        alpha = self.v_params(torch.cat([z_t, u_t], dim=1))
        M = self.A.shape[0]
        A = (alpha @ self.A.view(M, -1)).view(-1, self.dim_z, self.dim_z)
        B = (alpha @ self.B.view(M, -1)).view(-1, self.dim_z, self.dim_u)
        C = (alpha @ self.C.view(M, -1)).view(-1, self.dim_z, self.dim_w)
        return A, B, C

    def sample_w(self, x_t, z_t, u_t):
        data = torch.cat([x_t, z_t, u_t], dim=1)
        w_params = self.w_params(data)
        mean, std = torch.split(w_params, split_size_or_sections=self.dim_w, dim=1)
        std = torch.exp(std) + 1e-2
        return torch.normal(mean, std), dict(mean=mean, std=std)


    def fit(self, x, u):
        N, T, _ = x.shape
        z, w = torch.zeros((N, T, self.dim_z)), torch.zeros((N, T, self.dim_w))
        w_distributions = dict(mean=torch.zeros((N, T, self.dim_w)), std=torch.zeros((N, T, self.dim_w)))
        w_t, z_t, w_params = self.get_initial_samples(x)
        z[:, 0] = z_t
        w[:, 0] = w_t
        w_distributions['mean'][:, 0] = w_params['mean']
        w_distributions['std'][:, 0] = w_params['std']
        for t in range(1, T):
            u_t = u[:, t-1]
            w_t, w_params = self.sample_w(x[:, t], z_t, u_t)
            A, B, C = self.mix_matrices(z_t, u_t)
            z_t = (A @ z_t.unsqueeze(-1) + B @ u_t.unsqueeze(-1) + C @ w_t.unsqueeze(-1)).squeeze()
            z[:, t] = z_t
            w[:, t] = w_t.squeeze()
            w_distributions['mean'][:, t] = w_params['mean']
            w_distributions['std'][:, t] = w_params['std']

        x_rec_mean = self.observation_model(z).view(-1, self.dim_x)
        p_x = torch.distributions.MultivariateNormal(x_rec_mean, torch.diag(torch.ones(self.dim_x)))
        logprob_x = p_x.log_prob(x.view(-1, self.dim_x))

        w_mean, w_std = w_distributions['mean'].view(-1, self.dim_w), w_distributions['std'].view(-1, self.dim_w)
        q_w = torch.distributions.MultivariateNormal(w_mean, torch.diag_embed(w_std))
        prior_w = torch.distributions.MultivariateNormal(loc=torch.zeros_like(w_mean), covariance_matrix=torch.eye(self.dim_w))

        loss = logprob_x.sum() - torch.distributions.kl_divergence(q_w, prior_w).sum()
        return -loss





