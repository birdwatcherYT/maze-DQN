# coding: utf-8

import numpy as np


class Q_learning:
    """単純なQ-learning"""

    def __init__(self, maze):
        # 迷路
        self.maze = maze
        # 状態（=エージェントの位置）は迷路の盤面数
        row, col = self.maze.board_size()
        self.num_state = row * col
        # 行動の数は上下左右の4種類
        self.num_action = 4
        # Qは 状態数 x 行動数
        self.Q = np.zeros((self.num_state, self.num_action))
        # 現在の状態
        self.state = self.get_state()

    def select_best_action(self):
        """評価値の最も高い行動を探す"""
        return self.Q[self.state, :].argmax()

    def select_action(self, random_rate=0.01):
        """一定の確率で、ベストでない動きをする"""
        if np.random.rand() < random_rate:
            return np.random.randint(self.num_action)
        return self.select_best_action()

    def get_state(self):
        """状態を取得"""
        _, col = self.maze.board_size()
        x, y = self.maze.get_position()
        return x * col + y

    def reward(self):
        """報酬"""
        return 0 if self.maze.is_goal() else -1

    def from_start(self):
        """スタートからやり直す"""
        self.maze.reset()
        self.state = self.get_state()

    def step(self, learning_rate, discount_rate, random_rate):
        # 行動の選択。ベストアクションとは限らない。
        action = self.select_action(random_rate)
        # 選択された行動に従い動く。ただし、壁がある場合は無視される
        self.maze.move(action)
        # 移動後の状態を取得
        next_state = self.get_state()
        # ベストアクションを選択
        next_action = self.select_best_action()
        # Q[s][a] += 学習率 * ( 報酬 + 割引率 * ( max_{s'} Q[s'][a'] ) - Q[s][a] )
        self.Q[self.state][action] += learning_rate * (
            self.reward()
            + discount_rate * self.Q[next_state][next_action]
            - self.Q[self.state][action]
        )
        # 移動後の状態を現在の状態に記録
        self.state = next_state
