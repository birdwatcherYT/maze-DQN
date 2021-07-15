#!/usr/bin/env python
# coding: utf-8

import time
from Maze import Maze
from Deep_Q_learning import Deep_Q_learning

# 迷路
# W: Wall, S : Start, G: Goal
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
# 300episodeくらいでマシになる

# 何回ゴールするか
EPISODE_MAX = 1000
# ゴールまでの打ち切りステップ数
STEP_MAX = 3000
# 学習率
LEARNING_RATE = 0.001
# 割引率
DISCOUNT_RATE = 0.9
# 描画スピード
SLEEP_TIME = 0.0001

# 隠れ層のサイズ
HIDDEN_SIZE = 32
# 蓄える行動履歴データの数
MEMORY_SIZE = 500000
# バッチサイズ
BATCH_SIZE = 1024
# 損失関数
LOSS = "mse"
# LOSS = "huber_loss"
# ミニバッチで繰り返す学習回数
EPOCH = 2
# 更新する間隔
UPDATE_INTERVAL = 10
# ターゲットネットワークを更新する間隔
update_target_epsode_interval = 10
# 保存間隔
SAVE_INTERVAL = 10

maze = Maze(BOARD)
deep_q_learn = Deep_Q_learning(maze, LOSS, HIDDEN_SIZE, LEARNING_RATE, MEMORY_SIZE)

for episode in range(EPISODE_MAX):
    step = 0
    deep_q_learn.from_start()
    # ランダムに最善でない行動を取る
    random_rate = max(0.98 ** episode, 0.1)
    if episode % update_target_epsode_interval == 0:
        deep_q_learn.Qmain_to_Qtarget()
    while not maze.is_goal() and step < STEP_MAX:
        # エージェントの1ステップ(行動、評価値の更新)
        deep_q_learn.step(random_rate)
        # 迷路描画
        maze.draw()
        # Qネットワークの重みを更新する
        if step % UPDATE_INTERVAL == 0:
            deep_q_learn.update(BATCH_SIZE, DISCOUNT_RATE, EPOCH)
        step += 1
        time.sleep(SLEEP_TIME)

    if episode % SAVE_INTERVAL == 0:
        deep_q_learn.save_weights(f"model{episode}.h5")
    print("\x1b[K")  # 行末までをクリア
    print(f"episode : {episode} step : {step} ")

