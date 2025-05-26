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
PATH_WEIGH = "weights_parab_expert_2000.h5"
DEVICE = "cuda:0"
CONST = torch.tensor([[0, 5]], device=DEVICE)
GAMMA = 0.9
EPS = 0.2


def init_weight(layer) -> None:
    """
    Initialization of the layer's weights

    Args:
        layer : name of the layer

    """
    if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
        nn.init.normal_(layer.weight, 0, 0.02)
        nn.init.constant_(layer.bias, 0.01)
        print(f"Worked {layer}")


class NN(nn.Module):
    """
    Create NN
    Args:
        output_size: number of actions
        embedding_size: neurons in linear layer
        load_weight: True for used previous saved weights

    """

    def __init__(
        self,
        output_size: int = 2,
        embedding_size: int = 512,
        load_weight: bool = False,
    ):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(4, 32, 3, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, padding=1),
            nn.Conv2d(32, 64, 3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, padding=1),
            nn.Conv2d(64, 128, 3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, padding=1),
            nn.Conv2d(128, 128, 3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, padding=1),
        )

        self.pred = nn.Sequential(
            nn.Linear(4480, embedding_size),
            nn.ReLU(),
            nn.Linear(embedding_size, output_size),
        )

        if load_weight:
            self.work = self.load_state_dict(torch.load(PATH_WEIGH))
            print("weigts_loaded")
        else:
            self.model.apply(init_weight)
            self.pred.apply(init_weight)

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size()[0], -1)
        x = self.pred(x)
        x -= CONST
        return x


class Agent:
    """
    Create agent for plaing in flappy bird

    Args:
        max_memory: maximum number of records in dict (the default value 8000)
        learn_rate: learning rate of training (the default value 1e-5)
        load_weight: load weights of NN (the default value False)
        expert: decreas alpha or not (the default value False)
        learning: training mode or test (the default value False)
    """

    def __init__(
        self,
        max_memory: int = 8000,
        learn_rate: float = 1e-5,
        load_weight: bool = False,
        expert: bool = False,
        learning: bool = False,
    ):
        self.gamma = GAMMA
        self.memory = collections.deque(maxlen=max_memory)
        self.model = NN(load_weight=load_weight)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=learn_rate
        )
        self.expert = expert
        self.model.to(torch.device(DEVICE))
        self.epsilon = EPS
        self.alpha = 1
        self.learning = learning

    def remember(
        self,
        s: torch.tensor,
        a: torch.tensor,
        r: int,
        n_s: torch.tensor,
        target: torch.tensor,
        done: bool,
    ) -> None:
        """
        Add record to dict

        Args:
            s : system's state
            a : agent's action
            r : reward for action
            n_s : next systems state
            target : expert q_value
            done : lost gave or not

        """
        self.memory.append((s, a, r, n_s, target, done))

    def train_step(self, s, a, r, n_s, target, done) -> None:
        """
        Training NN

        Args:
            s : system's states
            a : agent's actions
            r : rewards for action
            n_s : next systems states
            target : expert q_values
            done : losts the games or not

        """
        self.criterion = nn.MSELoss()
        state = torch.cat(s) if type(s) is not torch.Tensor else s
        action = torch.cat(a) if type(a) is not torch.Tensor else a
        reward = torch.tensor(r, dtype=torch.float, device=DEVICE)
        next_state = torch.cat(n_s) if type(s) is not torch.Tensor else n_s
        target = torch.cat(target)
        pred = self.model(state)
        pred_next = self.model(next_state).detach()
        q_val = torch.sum(pred * action, dim=1)
        target_ql = torch.stack(
            tuple(
                (
                    reward[i]
                    if done[i]
                    else reward[i] + self.gamma * torch.max(pred_next[i])
                )
                for i in range(BATCH_SIZE)
            )
        )
        loss_q = self.criterion(q_val, target_ql)
        target_exper = torch.sum(target * action, dim=1)
        loss_exper = self.criterion(q_val, target_exper)
        if not self.expert:
            loss_exper = self.alpha * loss_exper
        self.model.eval()
        self.optimizer.zero_grad()
        loss = loss_q + loss_exper
        loss.backward()
        self.optimizer.step()

    def long_memory(self) -> None:
        """
        Create batch for training

        """
        if self.learning and len(self.memory) >= BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
            s, a, r, n_s, target, done = zip(*mini_sample)
            self.train_step(s, a, r, n_s, target, done)
