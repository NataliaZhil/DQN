# -*- coding: utf-8 -*-
"""
Created on Sun Jan 22 12:54:44 2023

@author: Natalia
"""

import torch
import torch.nn as nn
import random
import collections
import numpy as np

DEFAULT_VAL_DICT = 0
    
class Agent():
    def __init__(self, lr: float=0.001, load_q: bool = False, path= None):
        self.gamma=0.9
        self.lr=lr
        self.epsilon=0.5 if not load_q else 0.003
        self.Q={}
        self.last_jump=0
        if load_q:
            self.Q = np.load(path, allow_pickle='TRUE').item()
            print("load dict")
    def create_q(self, state):
        self.Q.setdefault(state, [DEFAULT_VAL_DICT, DEFAULT_VAL_DICT-0.001])
        return self.Q[state]
    
    def train_step(self, state, action, reward, next_state, done, jump): 
        self.Q.setdefault(next_state, [DEFAULT_VAL_DICT, DEFAULT_VAL_DICT-0.001])
        Q_new = self.lr*(reward + self.gamma*np.max(self.Q[next_state]) - self.Q[state][action[1]])
        self.Q[state][action[1]] += Q_new
        
        
