import copy
from ChessBoard import *


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
                if chess.team == self.team and chess.name in ["c", "m", "z","p"]:
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
            '''raise NotImplementedError(
                "Cannot determin next step!! Implement function ChessAI::min_max!!"
            )'''
            self.min_max(self.max_depth,chessboard)
            return self.old_pos + self.new_pos

        elif self.search_with == "alphabeta":
            '''raise NotImplementedError(
                "Cannot determin next step!! Implement function ChessAI::alpha_beta!!"
            )'''
            self.alpha_beta(self.max_depth, -float('inf'), float('inf'), chessboard)
            return self.old_pos + self.new_pos

    @staticmethod
    def get_nxt_player(player):
        if player == "r":
            return "b"
        else:
            return "r"


    '''def get_tmp_chessboard(chessboard, player_chess, new_row, new_col) -> ChessBoard:
        tmp_chessboard = copy.deepcopy(chessboard)
        tmp_chess = tmp_chessboard.chessboard_map[player_chess.row][player_chess.col]
        tmp_chess.row, tmp_chess.col = new_row, new_col
        tmp_chessboard.chessboard_map[new_row][new_col] = tmp_chess
        tmp_chessboard.chessboard_map[player_chess.row][player_chess.col] = None
        return tmp_chessboard'''

    '''@staticmethod
    def get_tmp_chessboard(chessboard, player_chess, new_row, new_col):
        """优化：直接操作现有棋盘，避免深拷贝"""
        # 1. 浅拷贝棋盘（仅复制棋子位置引用）
        tmp_chessboard = ChessBoard(screen=None)
        for row in range(10):
            for col in range(9):
                src_chess = chessboard.chessboard_map[row][col]
                if src_chess:
                    tmp_chessboard.chessboard_map[row][col] = Chess(
                        screen=None,
                        chess_name=f"{src_chess.team}_{src_chess.name}",
                        row=src_chess.row,
                        col=src_chess.col
                    )
                else:
                    tmp_chessboard.chessboard_map[row][col] = None

        # 2. 直接移动棋子（不重新创建对象）
        old_row, old_col = player_chess.row, player_chess.col
        tmp_chess = tmp_chessboard.chessboard_map[old_row][old_col]
        tmp_chess.row, tmp_chess.col = new_row, new_col
        tmp_chessboard.chessboard_map[new_row][new_col] = tmp_chess
        tmp_chessboard.chessboard_map[old_row][old_col] = None
        return tmp_chessboard'''

    def min_max(self, depth, chessboard: ChessBoard):
        def max_value(current_depth, current_player):
            if current_depth == 0 or chessboard.judge_win(current_player):
                return self.evaluate_class.evaluate(chessboard)

            max_eval = -float('inf')
            for chess in chessboard.get_chess():
                if chess.team != current_player:
                    continue
                old_row, old_col = chess.row, chess.col
                positions = chessboard.get_put_down_position(chess)
                for (new_row, new_col) in positions:
                    target_chess = chessboard.chessboard_map[new_row][new_col]
                    chessboard.chessboard_map[new_row][new_col] = chessboard.chessboard_map[old_row][old_col]
                    chessboard.chessboard_map[new_row][new_col].update_position(new_row, new_col)
                    chessboard.chessboard_map[old_row][old_col] = None

                    current_eval = min_value(current_depth - 1, self.get_nxt_player(current_player))

                    chessboard.chessboard_map[old_row][old_col] = chessboard.chessboard_map[new_row][new_col]
                    chessboard.chessboard_map[old_row][old_col].update_position(old_row, old_col)
                    chessboard.chessboard_map[new_row][new_col] = target_chess  # 还原被吃掉的棋子

                    max_eval = max(max_eval, current_eval)
            return max_eval

        def min_value(current_depth, current_player):
            if current_depth == 0 or chessboard.judge_win(current_player):
                return self.evaluate_class.evaluate(chessboard)

            min_eval = float('inf')
            for chess in chessboard.get_chess():
                if chess.team != current_player:
                    continue
                old_row, old_col = chess.row, chess.col
                positions = chessboard.get_put_down_position(chess)
                for (new_row, new_col) in positions:
                    target_chess = chessboard.chessboard_map[new_row][new_col]
                    chessboard.move_chess_silent(old_row, old_col, new_row, new_col)
                    current_eval = max_value(current_depth - 1, self.get_nxt_player(current_player))
                    chessboard.move_chess_silent(new_row, new_col, old_row, old_col)
                    chessboard.chessboard_map[new_row][new_col] = target_chess

                    min_eval = min(min_eval, current_eval)
            return min_eval

        # 主逻辑
        best_score = -float('inf')
        current_player = self.team
        for chess in chessboard.get_chess():
            if chess.team != current_player:
                continue
            old_row, old_col = chess.row, chess.col
            positions = chessboard.get_put_down_position(chess)
            for (new_row, new_col) in positions:
                target_chess = chessboard.chessboard_map[new_row][new_col]
                # 执行移动
                chessboard.move_chess_silent(old_row, old_col, new_row, new_col)
                # 评估当前局面
                score = min_value(depth - 1, self.get_nxt_player(current_player))
                # 回溯
                chessboard.move_chess_silent(new_row, new_col, old_row, old_col)
                chessboard.chessboard_map[new_row][new_col] = target_chess

                if score > best_score:
                    best_score = score
                    self.old_pos = [old_row, old_col]
                    self.new_pos = [new_row, new_col]
        return

    def alpha_beta(self, depth, alpha, beta, chessboard: ChessBoard):
        def max_value(current_depth, current_player, alpha, beta):
            if current_depth == 0 or chessboard.judge_win(current_player):
                return self.evaluate_class.evaluate(chessboard)

            max_value = -float('inf')
            # 按棋子价值排序（车、马等高价值棋子优先）
            chesses = sorted(
                [chess for chess in chessboard.get_chess() if chess.team == current_player],
                key=lambda x: self.evaluate_class.single_chess_point[x.name],
                reverse=True
            )
            for chess in chesses:
                positions = chessboard.get_put_down_position(chess)
                # 启发式排序：优先吃子或靠近对方将
                positions.sort(
                    key=lambda pos: self._move_heuristic(chess, pos, chessboard),
                    reverse=True
                )
                old_row,old_col = chess.row,chess.col

                for (new_row, new_col) in positions:

                    target_chess = chessboard.chessboard_map[new_row][new_col]
                    # 执行移动
                    chessboard.move_chess_silent(old_row, old_col, new_row, new_col)
                    # 评估当前局面
                    current_value = min_value(current_depth - 1, self.get_nxt_player(current_player), alpha,beta)
                    # 回溯
                    chessboard.move_chess_silent(new_row, new_col, old_row, old_col)
                    chessboard.chessboard_map[new_row][new_col] = target_chess

                    max_value = max(max_value,current_value)
                    if max_value >= beta:
                        return max_value  # Beta剪枝
                    alpha = max(alpha, max_value)
            return max_value

        def min_value(current_depth, current_player, alpha, beta):
            if current_depth == 0 or chessboard.judge_win(current_player):
                return self.evaluate_class.evaluate(chessboard)

            min_value = float('inf')
            chesses = sorted(
                [chess for chess in chessboard.get_chess() if chess.team == current_player],
                key=lambda x: self.evaluate_class.single_chess_point[x.name]
            )

            for chess in chesses:
                positions = chessboard.get_put_down_position(chess)
                positions.sort(
                    key=lambda pos: self._move_heuristic(chess, pos, chessboard)
                )
                old_row, old_col = chess.row, chess.col
                for (new_row, new_col) in positions:

                    target_chess = chessboard.chessboard_map[new_row][new_col]
                    # 执行移动
                    chessboard.move_chess_silent(old_row, old_col, new_row, new_col)
                    # 评估当前局面
                    current_value = max_value(current_depth - 1, self.get_nxt_player(current_player), alpha,beta)
                    # 回溯
                    chessboard.move_chess_silent(new_row, new_col, old_row, old_col)
                    chessboard.chessboard_map[new_row][new_col] = target_chess

                    min_value = min(min_value,current_value)
                    if min_value <= alpha:
                        return min_value  # Alpha剪枝
                    beta = min(beta, min_value)
            return min_value

        # 主逻辑：深度从初始值递减，优先探索高价值移动
        best_score = -float('inf')
        current_player = self.team
        chesses = sorted(
            [chess for chess in chessboard.get_chess() if chess.team == current_player],
            key=lambda x: self.evaluate_class.single_chess_point[x.name],
            reverse=True
        )

        #残局判断，避免对换车马炮导致和局
        if self._is_endgame(chessboard):
            depth += 1  # 增加搜索深度寻找杀棋

        for chess in chesses:
            if chess.name in ["c", "m", "p"]:
                positions = [pos for pos in chessboard.get_put_down_position(chess) if
                                not self._is_exchange_move(chess, pos, chessboard)]
            else:
                positions = chessboard.get_put_down_position(chess)

            positions.sort(
                key=lambda pos: self._move_heuristic(chess, pos, chessboard),
                reverse=True
            )
            old_row,old_col = chess.row,chess.col

            for (new_row, new_col) in positions:
                target_chess = chessboard.chessboard_map[new_row][new_col]
                chessboard.move_chess_silent(old_row, old_col, new_row, new_col)

                current_score = min_value(self.max_depth - 1, self.get_nxt_player(current_player), alpha , beta)

                chessboard.move_chess_silent(new_row, new_col, old_row, old_col)
                chessboard.chessboard_map[new_row][new_col] = target_chess

                if current_score > best_score:
                    best_score = current_score
                    self.old_pos = [old_row, old_col]
                    self.new_pos = [new_row, new_col]
                alpha = max(alpha, best_score)
                if alpha >= beta:
                    break
            if alpha >= beta:
                break

        return

    def _move_heuristic(self, chess, new_pos, chessboard):
        """启发式评分：吃子 > 靠近对方将 > 其他"""
        target_chess = chessboard.chessboard_map[new_pos[0]][new_pos[1]]
        if target_chess:
            return self.evaluate_class.single_chess_point[target_chess.name] * 3  # 吃子优先
        else:
            # 残局阶段：优先向对方将靠拢
            general_pos = chessboard.get_general_position("r" if chess.team == "b" else "b")
            if general_pos:
                distance = abs(new_pos[0] - general_pos[0]) + abs(new_pos[1] - general_pos[1])
                return -distance * 2  # 距离越小评分越高（权重加倍）
            else:
                return 0

    def _is_endgame(self, chessboard):
        """检测是否进入残局（对方棋子数 <= 2）"""
        opponent_pieces = [chess for chess in chessboard.get_chess() if chess.team != self.team]
        return len(opponent_pieces) <= 2

    def _is_exchange_move(self, chess, new_pos, chessboard):
        """判断移动是否为兑换（牺牲高价值棋子）"""
        target = chessboard.chessboard_map[new_pos[0]][new_pos[1]]
        return target and self.evaluate_class.single_chess_point[target.name] >= self.evaluate_class.single_chess_point[
            chess.name]

    def reset(self):
        self.old_pos = [-1, -1]
        self.new_pos = [0, 0]