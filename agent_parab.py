# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 13:31:11 2025

@author: n.zhilenkova
"""

import torch
import torch.nn as nn
#import pywt
import collections
import random
#import redone
import torch.nn.functional as F

BATCH_SIZE = 32
PATH_WEIGH = "weights_parab_True.h5"
DEVICE = "cuda:0"
CONST = torch.tensor([[0, 5]], device=DEVICE)

def init_weight(layer):
    if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
        nn.init.normal_(layer.weight, 0, 0.02)
        nn.init.constant_(layer.bias, 0.01)
        print(f"Worked {layer}")


class NN(nn.Module):
    def __init__(
        self,
        output_size: int = 2,
        embedding_size: int = 512,
        load_weight: bool = False,
    ):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(4, 32, 3, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, padding=1),
            nn.Conv2d(32, 64, 3, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, padding=1),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, padding=1),
  
        )

        self.pred = nn.Sequential(
            nn.Linear(6400, embedding_size),
            nn.ReLU(), 
            nn.Linear(embedding_size, output_size),
            #nn.Softmax()
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
        #print(x)
        x -= CONST
        #print(x)
        return x


class Agent:

    def __init__(
        self,
        max_memory: int = 8000,
        learn_rate: float = 1e-5,
        load_weight: bool = False,
        expert: bool = True,
    ):
        self.gamma = 0.9
        self.memory = collections.deque(maxlen=max_memory)
        self.model = NN(load_weight=load_weight)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=learn_rate
        )
        self.expert = expert
        
        self.model.to(torch.device(DEVICE))
        self.epsilon = 0.2
        

    def remember(self, s, a, r, n_s, target, done):
        self.memory.append((s, a, r, n_s, target, done))
            
    def train_step(self, s, a, r, n_s, target, done):
        self.criterion = nn.MSELoss() #if not self.expert else nn.CrossEntropyLoss() 
        state = torch.cat(s) if type(s) is not torch.Tensor else s
        action = torch.cat(a) if type(a) is not torch.Tensor else a
        reward = torch.tensor(r, dtype=torch.float, device=DEVICE)
        next_state = torch.cat(n_s) if type(s) is not torch.Tensor else n_s
        target = torch.cat(target) 
        pred = self.model(state)
        pred_next = self.model(next_state).detach()
        q_val = torch.sum(pred * action, dim=1)
        #if not self.expert:
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
            
    
        loss_exper = 0        
        loss_q = self.criterion(q_val, target_ql)
        if self.expert:
            target_exper = torch.sum(target * action, dim=1)
            loss_exper = self.criterion(q_val, target_exper) 
            #loss_q *= 0.4
        #print(loss_exper, loss_q)
        self.model.eval()
        self.optimizer.zero_grad()

        loss = loss_q + loss_exper
        #print(pred, loss, target_probs)
        loss.backward()
        self.optimizer.step()

    def long_memory(self):
        if len(self.memory) >= BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
            s, a, r, n_s, target,  done = zip(*mini_sample)    
            self.train_step(s, a, r, n_s, target, done)
        
