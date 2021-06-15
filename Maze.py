# coding: utf-8


class Maze:
    """迷路問題"""

    def __init__(self, board):
        self.board = board
        self.row = len(board)
        self.col = len(board[0])
        self.start_pos = None
        self.goal_pos = None
        for i in range(self.row):
            for j in range(self.col):
                if self.board[i][j] == "S":
                    self.start_pos = (i, j)
                elif self.board[i][j] == "G":
                    self.goal_pos = (i, j)
        self.agent_pos = list(self.start_pos)

    def reset(self):
        """エージェントをスタートに戻す"""
        self.agent_pos = list(self.start_pos)

    def get_position(self):
        """エージェントの位置を返す"""
        return tuple(self.agent_pos)

    def board_size(self):
        """盤面のサイズを返す"""
        return self.row, self.col

    # def get_around(self):
    #     x, y = self.agent_pos
    #     return (
    #         self.board[x - 1][y] != "W",
    #         self.board[x + 1][y] != "W",
    #         self.board[x][y - 1] != "W",
    #         self.board[x][y + 1] != "W",
    #     )

    def move(self, action: int):
        """ エージェントを移動させる。壁側への移動は無視される。

        Args:
            action (int): 0:上, 1:下, 2:左, 3:右
        """
        x, y = self.agent_pos
        if action == 0 and self.board[x - 1][y] != "W":
            self.agent_pos[0] -= 1
        elif action == 1 and self.board[x + 1][y] != "W":
            self.agent_pos[0] += 1
        elif action == 2 and self.board[x][y - 1] != "W":
            self.agent_pos[1] -= 1
        elif action == 3 and self.board[x][y + 1] != "W":
            self.agent_pos[1] += 1

    def is_goal(self) -> bool:
        """ゴールしたかどうか

        Returns:
            bool: ゴール位置にいたらTrue
        """
        x, y = self.agent_pos
        return self.board[x][y] == "G"

    def draw(self):
        """盤面を描画"""
        print("\x1b[0;0H")  # 画面クリア
        for i in range(self.row):
            for j in range(self.col):
                print(self.board[i][j] if [i, j] != self.agent_pos else "A", end="")
            print(" ")

