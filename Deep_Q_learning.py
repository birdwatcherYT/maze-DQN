# coding: utf-8

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.layers.advanced_activations import PReLU
from collections import deque


class Memory:
    def __init__(self, max_size):
        self.memory = deque(maxlen=max_size)

    def add(self, experience):
        self.memory.append(experience)

    def sample(self, batch_size):
        select = np.random.choice(self.len(), size=batch_size, replace=False)
        return [self.memory[i] for i in select]

    def len(self):
        return len(self.memory)


def DQN(loss, input_dim, output_dim, hidden_size, learning_rate):
    model = Sequential()
    model.add(Dense(hidden_size, input_dim=input_dim))
    model.add(PReLU())
    model.add(Dense(hidden_size))
    model.add(PReLU())
    model.add(Dense(output_dim))
    model.compile(optimizer=Adam(learning_rate), loss=loss)
    return model


class Deep_Q_learning:
    """Deep Q-learning"""

    def __init__(self, maze, loss, hidden_size, learning_rate, memory_size):
        self.maze = maze
        # 状態の次元数は、(i, j)座標の2次元
        self.state_dim = 2
        # 行動の数は上下左右の4種類
        self.num_action = 4
        # 現在の状態
        self.state = self.get_state()
        # Q が NNになる
        self.Qmain = DQN(
            loss, self.state_dim, self.num_action, hidden_size, learning_rate
        )
        self.Qtarget = DQN(
            loss, self.state_dim, self.num_action, hidden_size, learning_rate
        )
        # メモリ
        self.memory = Memory(memory_size)

    def update(self, batch_size, discount_rate, epoch):
        if self.memory.len() < batch_size:
            return
        mini_batch = self.memory.sample(batch_size)

        # 状態を入力として
        inputs = np.array([state for (state, _, _, _, _) in mini_batch])
        # 基本はQネットワークの出力をそのまま予測する
        targets = self.Qmain.predict(inputs)

        actions = np.array([action for (_, action, _, _, _) in mini_batch])
        rewards = np.array([reward for (_, _, reward, _, _) in mini_batch])

        next_states = np.array([next_state for (_, _, _, next_state, _) in mini_batch])
        not_fins = np.array([int(not fin) for (_, _, _, _, fin) in mini_batch])
        next_actions = self.Qmain.predict(next_states).argmax(axis=1)
        pred_max = self.Qtarget.predict(next_states)[
            np.arange(batch_size), next_actions
        ]
        # 最小化したいものは E[ ( 報酬 + 割引率 * ( max_{a'} Q(s', a') ) - Q(s, a) )^2 ] なので、
        # 報酬 + 割引率 * ( max_{a'} Q(s', a') ) がターゲット
        targets[np.arange(batch_size), actions] = rewards + discount_rate * pred_max * not_fins

        self.Qmain.fit(inputs, targets, epochs=epoch, verbose=0)

    def select_action(self, random_rate):
        """一定の確率で、ベストでない動きをする"""
        if np.random.rand() < random_rate:
            return np.random.randint(self.num_action)
        return self.select_best_action()

    def select_best_action(self):
        """評価値の最も高い行動を探す"""
        pred = self.Qmain.predict([self.state])[0]
        action = pred.argmax()
        return action

    def get_state(self):
        """状態を取得"""
        row, col = self.maze.board_size()
        x, y = self.maze.get_position()
        return x / row, y / col
        # return x, y

    def reward(self):
        """報酬"""
        return 1 if self.maze.is_goal() else -1
        # return 0 if self.maze.is_goal() else -1
        # return 1 if self.maze.is_goal() else 0

    def from_start(self):
        """スタートからやり直す"""
        self.maze.reset()
        self.state = self.get_state()

    def step(self, random_rate):
        # 行動の選択。ベストアクションとは限らない。
        action = self.select_action(random_rate)
        # 選択された行動に従い動く。ただし、壁がある場合は無視される
        self.maze.move(action)
        # 移動後の状態を取得
        next_state = self.get_state()
        # ゴールしたかどうか
        fin = self.maze.is_goal()
        # 報酬
        reward = self.reward()
        # メモリに追加
        self.memory.add((self.state, action, reward, next_state, fin))
        # 移動後の状態を現在の状態に記録
        self.state = next_state

    def Qmain_to_Qtarget(self):
        self.Qtarget.set_weights(self.Qmain.get_weights())

    def save_weights(self, filename):
        self.Qmain.save_weights(filename)

    def load_weights(self, filename):
        self.Qmain.load_weights(filename)
        self.Qmain_to_Qtarget()
