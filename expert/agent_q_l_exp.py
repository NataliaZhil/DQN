# -*- coding: utf-8 -*-
"""
Created on Sun Jan 22 12:54:44 2023

@author: Natalia
"""


import numpy as np

DEFAULT_VAL_DICT = 0
GAMMA = 0.9
START_EPS = (0.5, 0.003)
PENALTY = 0.001


class Agent:
    """
    Agent for plaing in flappy bird. Training with q-learning

    Args:
        lr: learning rate (the default is 0.001)
        learning: True for training value func
        load_q: load dict of q-values (the default is False)
        path: path to saved dict, used if load_q = True
            (the default is "weights_exper_ql.npy")

    """

    def __init__(
        self,
        lr: float = 0.001,
        learning: bool = False,
        load_q: bool = False,
        path: str = "weights_exper_ql.npy",
    ) -> None:

        self.gamma = GAMMA
        self.learning = learning
        self.lr = lr
        self.epsilon = START_EPS[0] if not load_q else START_EPS[1]
        self.Q = {}
        if load_q:
            self.Q = np.load(path, allow_pickle="TRUE").item()
            print("load dict")

    def create_q(self, state: tuple) -> list:
        """
        Set value for new keys in Q and return
        q-values for each action

        Args:
            state : state of the game

        Returns:
            list of q-values according to the state

        """
        self.Q.setdefault(
            state, [DEFAULT_VAL_DICT, DEFAULT_VAL_DICT - PENALTY]
        )
        return self.Q[state]

    def train_step(
        self,
        state: tuple,
        action: np.ndarray,
        reward: float,
        next_state: tuple,
        done: bool,
    ) -> None:
        """
        Estimates q-values

        Args:
            state : state of the game
            action : array of actions
            reward : reward for the action
            next_state : state of the game after making the action
            done : defune lost game or not

        """
        if self.learning:
            self.Q.setdefault(
                next_state, [DEFAULT_VAL_DICT, DEFAULT_VAL_DICT - PENALTY]
            )
            Q_new = self.lr * (
                reward
                + self.gamma * np.max(self.Q[next_state])
                - self.Q[state][action[1]]
            )
            self.Q[state][action[1]] += Q_new
