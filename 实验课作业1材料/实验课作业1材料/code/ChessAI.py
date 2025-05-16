import copy
from ChessBoard import *
import time


class Evaluate(object):
    # 棋子棋力得分
    single_chess_point = {
        "c": 989,  # 车
        "m": 439,  # 马
        "p": 442,  # 炮
        "s": 226,  # 士
        "x": 210,  # 象
        "z": 55,  # 卒
        "j": 65536,  # 将
    }
    # 红兵（卒）位置得分
    red_bin_pos_point = [
        [1, 3, 9, 10, 12, 10, 9, 3, 1],
        [18, 36, 56, 95, 118, 95, 56, 36, 18],
        [15, 28, 42, 73, 80, 73, 42, 28, 15],
        [13, 22, 30, 42, 52, 42, 30, 22, 13],
        [8, 17, 18, 21, 26, 21, 18, 17, 8],
        [3, 0, 7, 0, 8, 0, 7, 0, 3],
        [-1, 0, -3, 0, 3, 0, -3, 0, -1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
    # 红车位置得分
    red_che_pos_point = [
        [185, 195, 190, 210, 220, 210, 190, 195, 185],
        [185, 203, 198, 230, 245, 230, 198, 203, 185],
        [180, 198, 190, 215, 225, 215, 190, 198, 180],
        [180, 200, 195, 220, 230, 220, 195, 200, 180],
        [180, 190, 180, 205, 225, 205, 180, 190, 180],
        [155, 185, 172, 215, 215, 215, 172, 185, 155],
        [110, 148, 135, 185, 190, 185, 135, 148, 110],
        [100, 115, 105, 140, 135, 140, 105, 115, 110],
        [115, 95, 100, 155, 115, 155, 100, 95, 115],
        [20, 120, 105, 140, 115, 150, 105, 120, 20],
    ]
    # 红马位置得分
    red_ma_pos_point = [
        [80, 105, 135, 120, 80, 120, 135, 105, 80],
        [80, 115, 200, 135, 105, 135, 200, 115, 80],
        [120, 125, 135, 150, 145, 150, 135, 125, 120],
        [105, 175, 145, 175, 150, 175, 145, 175, 105],
        [90, 135, 125, 145, 135, 145, 125, 135, 90],
        [80, 120, 135, 125, 120, 125, 135, 120, 80],
        [45, 90, 105, 190, 110, 90, 105, 90, 45],
        [80, 45, 105, 105, 80, 105, 105, 45, 80],
        [20, 45, 80, 80, -10, 80, 80, 45, 20],
        [20, -20, 20, 20, 20, 20, 20, -20, 20],
    ]
    # 红炮位置得分
    red_pao_pos_point = [
        [190, 180, 190, 70, 10, 70, 190, 180, 190],
        [70, 120, 100, 90, 150, 90, 100, 120, 70],
        [70, 90, 80, 90, 200, 90, 80, 90, 70],
        [60, 80, 60, 50, 210, 50, 60, 80, 60],
        [90, 50, 90, 70, 220, 70, 90, 50, 90],
        [120, 70, 100, 60, 230, 60, 100, 70, 120],
        [10, 30, 10, 30, 120, 30, 10, 30, 10],
        [30, -20, 30, 20, 200, 20, 30, -20, 30],
        [30, 10, 30, 30, -10, 30, 30, 10, 30],
        [20, 20, 20, 20, -10, 20, 20, 20, 20],
    ]
    # 红将位置得分
    red_jiang_pos_point = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 9750, 9800, 9750, 0, 0, 0],
        [0, 0, 0, 9900, 9900, 9900, 0, 0, 0],
        [0, 0, 0, 10000, 10000, 10000, 0, 0, 0],
    ]
    # 红相或士位置得分
    red_xiang_shi_pos_point = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 60, 0, 0, 0, 60, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [80, 0, 0, 80, 90, 80, 0, 0, 80],
        [0, 0, 0, 0, 0, 120, 0, 0, 0],
        [0, 0, 70, 100, 0, 100, 70, 0, 0],
    ]

    red_pos_point = {
        "z": red_bin_pos_point,
        "m": red_ma_pos_point,
        "c": red_che_pos_point,
        "j": red_jiang_pos_point,
        "p": red_pao_pos_point,
        "x": red_xiang_shi_pos_point,
        "s": red_xiang_shi_pos_point,
    }

    def __init__(self, team):
        self.team = team

    def get_single_chess_point(self, chess: Chess):
        if chess.team == self.team:
            return self.single_chess_point[chess.name]
        else:
            return -1 * self.single_chess_point[chess.name]

    def get_chess_pos_point(self, chess: Chess):
        red_pos_point_table = self.red_pos_point[chess.name]
        if chess.team == "r":
            pos_point = red_pos_point_table[chess.row][chess.col]
        else:
            pos_point = red_pos_point_table[9 - chess.row][chess.col]
        if chess.team != self.team:
            pos_point *= -1
        return pos_point

    def evaluate(self, chessboard: ChessBoard):
        point = 0
        general_pos = None  # 对方将的位置
        for chess in chessboard.get_chess():
            # 基础分值和位置分
            point += self.get_single_chess_point(chess)
            point += self.get_chess_pos_point(chess)
            # 记录对方将的位置
            if chess.name == "j" and chess.team != self.team:
                general_pos = (chess.row, chess.col)

        # 残局激励：如果我方有cmzp，且对方将暴露，增加围剿分
        if general_pos:
            attacker_count = 0  # 我方可攻击对方将的棋子数
            for chess in chessboard.get_chess():
                if chess.team == self.team and chess.name in ["c", "m", "z", "p"]:
                    # 判断是否在对方将的攻击范围内
                    positions = chessboard.get_put_down_position(chess)
                    if general_pos in positions:
                        attacker_count += 1
            point += attacker_count * 200  # 每个可攻击将的棋子加200分

        opponent_general = None
        for chess in chessboard.get_chess():
            if chess.name == "j" and chess.team != self.team:
                opponent_general = chess
                break
        if opponent_general:
            movable_positions = chessboard.get_put_down_position(opponent_general)
            freedom_penalty = -len(movable_positions) * 100  # 对方将可移动位置越少，我方得分越高
            point += freedom_penalty

            # 棋子协同奖励（车、马、炮与其他棋子配合）
        for chess in chessboard.get_chess():
            if chess.team == self.team and chess.name in ["c", "m", "p"]:
                nearby_allies = 0
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    x, y = chess.row + dx, chess.col + dy
                    if 0 <= x < 10 and 0 <= y < 9:
                        if chessboard.chessboard_map[x][y] and chessboard.chessboard_map[x][y].team == self.team:
                            nearby_allies += 1
                point += nearby_allies * 50  # 每有一个友方相邻，加50分

        return point


class ChessMap(object):
    def __init__(self, chessboard: ChessBoard):
        self.chess_map = copy.deepcopy(chessboard.chessboard_map)


class ChessAI(object):

    def reset(self):
        pass

    def __init__(self, computer_team, max_depth=5, search_with="minmax"):
        self.team = computer_team
        self.max_depth = max_depth
        self.old_pos = [-1, -1]
        self.new_pos = [0, 0]
        self.evaluate_class = Evaluate(self.team)
        self.search_with = search_with

    def get_next_step(self, chessboard: ChessBoard):
        """
        该函数应当返回四个值:
            1 要操作棋子的横坐标
            2 要操作棋子的纵坐标
            3 落子的横坐标
            4 落子的纵坐标
        """

        if self.search_with == "minmax":
            self.record = []
            st = time.time()
            score = self.min_max(0, chessboard)
            # print(time.time()-st)
            for rec in self.record:
                if rec[3] == score:
                    return (rec[0].row, rec[0].col, rec[1], rec[2])
            raise NotImplementedError(
                "Cannot determin next step!! Implement function ChessAI::min_max!!"
            )
        elif self.search_with == "alphabeta":
            self.record = []
            st = time.time()
            score = self.alpha_beta(0, -1e10, 1e10, chessboard)
            # print(time.time()-st)
            for rec in self.record:
                if rec[3] == score:
                    return (rec[0].row, rec[0].col, rec[1], rec[2])
            raise NotImplementedError(
                "Cannot determin next step!! Implement function ChessAI::alpha_beta!!"
            )

    def get_successor(self, chessboard: ChessBoard):
        ans = []
        l = chessboard.get_chess()
        for chess in l:
            pla = chessboard.get_put_down_position(chess)
            for x, y in pla:
                ans.append((chess, x, y))
        return ans

    @staticmethod
    def get_nxt_player(player):
        if player == "r":
            return "b"
        else:
            return "r"

    @staticmethod
    def get_tmp_chessboard(chessboard, player_chess, new_row, new_col) -> ChessBoard:
        tmp_chessboard = copy.deepcopy(chessboard)
        tmp_chess = tmp_chessboard.chessboard_map[player_chess.row][player_chess.col]
        tmp_chess.row, tmp_chess.col = new_row, new_col
        tmp_chessboard.chessboard_map[new_row][new_col] = tmp_chess
        tmp_chessboard.chessboard_map[player_chess.row][player_chess.col] = None
        return tmp_chessboard

    def min_max(self, depth, chessboard: ChessBoard):
        if (chessboard.judge_win(self.team) or chessboard.judge_win(
                self.get_nxt_player(self.team)) or depth == self.max_depth):
            return self.evaluate_class.evaluate(chessboard)
        if (depth % 2 == 0):
            result = -1e10
            for chess, x, y in self.get_successor(chessboard):
                if chess.team == self.team:

                    old_row, old_col = chess.row, chess.col
                    position_chess_backup = chessboard.chessboard_map[x][y]
                    chessboard.chessboard_map[x][y] = chessboard.chessboard_map[old_row][old_col]
                    chessboard.chessboard_map[x][y].update_position(x, y)
                    chessboard.chessboard_map[old_row][old_col] = None

                    result = max(result, self.min_max(depth + 1, chessboard))

                    chessboard.chessboard_map[old_row][old_col] = chessboard.chessboard_map[x][y]
                    chessboard.chessboard_map[old_row][old_col].update_position(old_row, old_col)
                    chessboard.chessboard_map[x][y] = position_chess_backup

                    if depth == 0:
                        self.record.append((chess, x, y, result))

            return result
        else:
            result = 1e10
            for chess, x, y in self.get_successor(chessboard):
                if chess.team != self.team:
                    old_row, old_col = chess.row, chess.col
                    position_chess_backup = chessboard.chessboard_map[x][y]
                    chessboard.chessboard_map[x][y] = chessboard.chessboard_map[old_row][old_col]
                    chessboard.chessboard_map[x][y].update_position(x, y)
                    chessboard.chessboard_map[old_row][old_col] = None

                    result = min(result, self.min_max(depth + 1, chessboard))

                    chessboard.chessboard_map[old_row][old_col] = chessboard.chessboard_map[x][y]
                    chessboard.chessboard_map[old_row][old_col].update_position(old_row, old_col)
                    chessboard.chessboard_map[x][y] = position_chess_backup
                    # chessboard.move_chess_silent(x,y,chess.row,chess.col)
            return result
        raise NotImplementedError(
            "Cannot determin next step!! Implement function ChessAI::min_max!!"
        )

    def alpha_beta(self, depth, a, b, chessboard: ChessBoard):
        if (chessboard.judge_win(self.team) or chessboard.judge_win(
                self.get_nxt_player(self.team)) or depth == self.max_depth):
            return self.evaluate_class.evaluate(chessboard)
        if (depth % 2 == 0):
            a = -1e10
            for chess, x, y in self.get_successor(chessboard):
                if chess.team == self.team:

                    old_row, old_col = chess.row, chess.col
                    position_chess_backup = chessboard.chessboard_map[x][y]
                    chessboard.chessboard_map[x][y] = chessboard.chessboard_map[old_row][old_col]
                    chessboard.chessboard_map[x][y].update_position(x, y)
                    chessboard.chessboard_map[old_row][old_col] = None

                    a = max(a, self.alpha_beta(depth + 1, a, b, chessboard))

                    chessboard.chessboard_map[old_row][old_col] = chessboard.chessboard_map[x][y]
                    chessboard.chessboard_map[old_row][old_col].update_position(old_row, old_col)
                    chessboard.chessboard_map[x][y] = position_chess_backup

                    if depth == 0:
                        self.record.append((chess, x, y, a))

                    if (a > b):
                        return a

            return a
        else:
            b = 1e10
            for chess, x, y in self.get_successor(chessboard):
                if chess.team != self.team:

                    old_row, old_col = chess.row, chess.col
                    position_chess_backup = chessboard.chessboard_map[x][y]
                    chessboard.chessboard_map[x][y] = chessboard.chessboard_map[old_row][old_col]
                    chessboard.chessboard_map[x][y].update_position(x, y)
                    chessboard.chessboard_map[old_row][old_col] = None

                    b = min(b, self.alpha_beta(depth + 1, a, b, chessboard))

                    chessboard.chessboard_map[old_row][old_col] = chessboard.chessboard_map[x][y]
                    chessboard.chessboard_map[old_row][old_col].update_position(old_row, old_col)
                    chessboard.chessboard_map[x][y] = position_chess_backup
                    # chessboard.move_chess_silent(x,y,chess.row,chess.col)
                    if (a > b):
                        return a
            return b
        raise NotImplementedError(
            "Cannot determin next step!! Implement function ChessAI::alpha_beta!!"
        )
