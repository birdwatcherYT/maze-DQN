#!/usr/bin/env python
# coding: utf-8

# # 強化学習入門 実践

import time

from Maze import Maze
from Deep_Q_learning import Deep_Q_learning

# 迷路
# W: Wall, S : Start, G: Goal
# BOARD = [
#     "WWWWWWWWWWWWWWWWWWWWW",
#     "WS  W W         W  GW",
#     "WWW W W W W W W W W W",
#     "W       W W W W   W W",
#     "WWW WWW WWW WWWWW W W",
#     "W   W W W         W W",
#     "WWW W W WWWWWWW WWW W",
#     "W                   W",
#     "WWWWWWWWWWWWWWWWWWWWW",
# ]
BOARD = [
    "WWWWWWWWWWWWWWWWWWWWW",
    "WS    W     W       W",
    "W W WWWWW W W W WWWWW",
    "W W       W   W W   W",
    "W W W WWW WWWWW W WWW",
    "W W W       W       W",
    "W W W WWW WWW W WWWWW",
    "W             W    GW",
    "WWWWWWWWWWWWWWWWWWWWW",
]

# 何回ゴールするか
EPISODE_MAX = 1000
# ゴールまでの打ち切りステップ数
STEP_MAX = 3000
# 学習率
LEARNING_RATE = 0.0001
# 割引率
DISCOUNT_RATE = 0.95
# 描画スピード
SLEEP_TIME = 0.0001

# 隠れ層のサイズ
HIDDEN_SIZE = 32
# 蓄える行動履歴データの数
MEMORY_SIZE = 10000
# バッチサイズ
BATCH_SIZE = 32
# 損失関数
LOSS = "huber_loss"


maze = Maze(BOARD)
deep_q_learn = Deep_Q_learning(maze, LOSS, HIDDEN_SIZE, LEARNING_RATE, MEMORY_SIZE)

# NOTE: まだ検討中
update_interval = 50
save_interval = 5
# deep_q_learn.Qmodel.load_weights(f"episode{episode}.hdf5")

for episode in range(EPISODE_MAX):
    step = 0
    deep_q_learn.from_start()
    # ランダムに最善でない行動を取る
    random_rate = 0.1 + 0.9 / (1 + episode)
    # random_rate = 0.1 + 0.95 ** (episode)
    deep_q_learn.Qmain_to_Qtarget()
    while not maze.is_goal() and step < STEP_MAX:
        # エージェントの1ステップ(行動、評価値の更新)
        deep_q_learn.step(DISCOUNT_RATE, random_rate, BATCH_SIZE)
        # 迷路描画
        maze.draw()
        step += 1
        # time.sleep(SLEEP_TIME)

        # Qネットワークの重みを更新する
        if step % update_interval == 0:
            deep_q_learn.update(BATCH_SIZE, DISCOUNT_RATE)
    # Qネットワークの重みを更新する
    # deep_q_learn.update(BATCH_SIZE, DISCOUNT_RATE)
    # if episode % save_interval == 0:
    #     deep_q_learn.Qmodel.save_weights(f"episode{episode}.hdf5")
    print("\x1b[K")  # 行末までをクリア
    print(f"episode : {episode} step : {step} ")

