# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 13:31:11 2025

@author: n.zhilenkova
"""

import torch
import torch.nn as nn
import collections
import random


BATCH_SIZE = 32
PATH_WEIGH = "weights_jump_strait.h5"
DEVICE = "cuda:0"


def init_weight(layer):
    if isinstance(layer, nn.Linear):
        nn.init.normal_(layer.weight, 0, 0.02)
        nn.init.constant_(layer.bias, 0.01)
        print(f"Worked {layer}")


class NN(nn.Module):
    def __init__(
        self,
        output_size: int = 2,
        input_layer: int = 4,
        hidden_layer: int = 256,
        load_weight: bool = False,
        path=None,
    ):
        super().__init__()
        self.pred = nn.Sequential(
            nn.Linear(input_layer, hidden_layer),
            #nn.ReLU(),
            nn.Linear(hidden_layer, hidden_layer),
            #nn.ReLU(),
            nn.Linear(hidden_layer, 2),
        )

        if load_weight:
            self.work = self.load_state_dict(
                torch.load(PATH_WEIGH, map_location=torch.device(DEVICE))
            )
            print("weigts_loaded")
        else:
            self.pred.apply(init_weight)

    def forward(self, x):
        x = self.pred(x)
        return x


class Agent:

    def __init__(
        self,
        max_memory: int = 10000,
        learn_rate: float = 1e-4,
        load_weight: bool = False,
        path=None,
    ):
        self.gamma = 0.9
        self.memory = collections.deque(maxlen=max_memory)
        self.model = NN(load_weight=load_weight)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=learn_rate
        )
        self.criterion = nn.MSELoss()
        self.model.to(torch.device(DEVICE))
        self.epsilon = 0.02 if not load_weight else 0

    def remember(self, s, a, r, n_s, done):
        self.memory.append((s, a, r, n_s, done))

    def train_step(self, s, a, r, n_s, done):
        state = torch.cat(s) if type(s) is not torch.Tensor else s
        action = torch.cat(a) if type(a) is not torch.Tensor else a
        reward = torch.tensor(r, dtype=torch.float, device=DEVICE)
        next_state = torch.cat(n_s) if type(s) is not torch.Tensor else n_s
        pred = self.model(state)
        pred_next = self.model(next_state).detach()
        target = torch.stack(
            tuple(
                (
                    reward[i]
                    if done[i]
                    else reward[i] + self.gamma * torch.max(pred_next[i])
                )
                for i in range(BATCH_SIZE)
            )
        )
        

        q_val = torch.sum(pred * action, dim=1)
        #self.model.eval()
        self.optimizer.zero_grad()

        loss = self.criterion(target, q_val)
        loss.backward()
        self.optimizer.step()

    def long_memory(self):
        if len(self.memory) >= BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
            s, a, r, n_s, done = zip(*mini_sample)
            self.train_step(s, a, r, n_s, done)

