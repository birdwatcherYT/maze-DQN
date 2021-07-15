#!/usr/bin/env python
# coding: utf-8

import time
from Maze import Maze
from Q_learning import Q_learning

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

# 何回ゴールするか
EPISODE_MAX = 1000
# ゴールまでの打ち切りステップ数
STEP_MAX = 3000
# 学習率
LEARNING_RATE = 0.1
# 割引率
DISCOUNT_RATE = 0.95
# 描画スピード
SLEEP_TIME = 0.001

maze = Maze(BOARD)
q_learn = Q_learning(maze)

for episode in range(EPISODE_MAX):
    step = 0
    q_learn.from_start()
    # ランダムに最善でない行動を取る
    random_rate = 0.01 + 0.9 / (1 + episode)
    while not maze.is_goal() and step < STEP_MAX:
        # エージェントの1ステップ(行動、評価値の更新)
        q_learn.step(LEARNING_RATE, DISCOUNT_RATE, random_rate)
        # 迷路描画
        maze.draw()
        step += 1
        time.sleep(SLEEP_TIME)
    print("\x1b[K")  # 行末までをクリア
    print(f"episode : {episode} step : {step} ")
