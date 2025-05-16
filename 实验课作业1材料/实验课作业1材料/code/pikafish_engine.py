import subprocess
import threading
import sys


class PikafishEngine:
    """
    A class to interface with the Pikafish chess engine for AI-based move generation.
    """

    def __init__(
        self,
        team,
        eval_path="../Pikafish-20250110/pikafish.nnue",
        max_depth=1,
        skill_level=0,
        multi_pv=1,
    ):
        """
        Initializes the Pikafish engine with the specified parameters.

        Args:
            team (str): The team color ('r' for red, 'b' for black).
            eval_path (str): Path to the evaluation file for the engine.
            max_depth (int): Maximum search depth for the engine.
            skill_level (int): Skill level of the engine.
            multi_pv (int): Number of principal variations to calculate.
        """
        # Determine the engine executable path based on the operating system
        if sys.platform.startswith("win"):
            engine_path = "../Pikafish-20250110/pikafish-bmi2.exe"
        elif sys.platform.startswith("darwin"):
            engine_path = "../Pikafish-20250110/MacOS/pikafish-apple-silicon"
        else:
            engine_path = "../Pikafish-20250110/Linux/pikafish-bmi2"

        # Set the team color and maximum search depth
        self.team = "w" if team == "r" else "b"
        self.max_depth = max_depth

        # Start the engine process
        self.process = subprocess.Popen(
            engine_path,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        )
        self.output = []  # Stores engine output
        self.running = True  # Indicates if the engine is running
        self.lock = threading.Lock()  # Lock for thread-safe operations
        self.bestmove_event = threading.Event()  # Event to signal when a best move is found
        self.bestmove = None  # Stores the best move found by the engine
        self._start_reader_thread()  # Start a thread to read engine output

        # Initialize the engine with UCI protocol and set options
        self.send_command("uci")
        self.set_eval_file(eval_path)
        self.send_command(f"setoption name Skill Level value {skill_level}")
        self.send_command(f"setoption name MultiPV value {multi_pv}")

        # Mapping custom piece notation to UCCI FEN characters
        self.piece_map = {
            "b_c": "r",  # Black Rook
            "b_m": "n",  # Black Knight
            "b_x": "b",  # Black Bishop
            "b_s": "a",  # Black Advisor
            "b_j": "k",  # Black King
            "b_p": "c",  # Black Cannon
            "b_z": "p",  # Black Pawn
            "r_c": "R",  # Red Rook
            "r_m": "N",  # Red Knight
            "r_x": "B",  # Red Bishop
            "r_s": "A",  # Red Advisor
            "r_j": "K",  # Red King
            "r_p": "C",  # Red Cannon
            "r_z": "P",  # Red Pawn
        }

    def _start_reader_thread(self):
        """
        Starts a thread to continuously read output from the engine.
        """
        def read_output():
            while self.running:
                line = self.process.stdout.readline()
                if line:
                    line = line.strip()
                    # Append the output to the list and set the best move event if applicable
                    with self.lock:
                        self.output.append(line)
                        if line.startswith("bestmove"):
                            self.bestmove = line.split()[1]
                            self.bestmove_event.set()

        threading.Thread(target=read_output, daemon=True).start()

    def send_command(self, command):
        """
        Sends a command to the engine.

        Args:
            command (str): The command to send.
        """
        self.process.stdin.write(command + "\n")
        self.process.stdin.flush()

    def set_eval_file(self, path):
        """
        Sets the evaluation file for the engine.

        Args:
            path (str): Path to the evaluation file.
        """
        self.send_command(f"setoption name EvalFile value {path}")

    def set_position(self, fen):
        """
        Sets the board position using a FEN string.

        Args:
            fen (str): The FEN string representing the board position.
        """
        self.send_command(f"position fen {fen}")

    def go(self):
        """
        Starts the engine's search for the best move.
        """
        self.bestmove = None
        self.bestmove_event.clear()
        self.send_command("go")

    def parse_chessboard_to_fenstr(self, chessboard):
        """
        Converts the custom chessboard representation to a UCCI FEN string.

        Args:
            chessboard (ChessBoard): The chessboard object.

        Returns:
            str: The FEN string representation of the board.
        """
        board = chessboard.get_chessboard_str_map()

        fen_ranks = []
        for rank in board:
            fen_rank = []
            empty_count = 0
            for square in rank:
                if square == "":
                    empty_count += 1
                else:
                    if empty_count > 0:
                        fen_rank.append(str(empty_count))
                        empty_count = 0
                    fen_rank.append(self.piece_map[square])
            if empty_count > 0:
                fen_rank.append(str(empty_count))
            fen_ranks.append("".join(fen_rank))

        return "/".join(fen_ranks)

    def ucci_move_to_coords(self, move_str):
        """
        Converts a UCCI move string (e.g., h2e2) to board coordinates.

        Args:
            move_str (str): The UCCI move string.

        Returns:
            tuple: A tuple containing (cur_row, cur_col, nxt_row, nxt_col).
        """
        if len(move_str) != 4:
            raise ValueError(f"Invalid move format: {move_str}")

        file_map = {
            "a": 0,
            "b": 1,
            "c": 2,
            "d": 3,
            "e": 4,
            "f": 5,
            "g": 6,
            "h": 7,
            "i": 8,
        }

        from_file = file_map[move_str[0]]
        from_rank = int(move_str[1])
        to_file = file_map[move_str[2]]
        to_rank = int(move_str[3])

        # Convert to board coordinates
        cur_row = 9 - from_rank
        cur_col = from_file
        nxt_row = 9 - to_rank
        nxt_col = to_file

        return (cur_row, cur_col, nxt_row, nxt_col)

    def get_next_step(self, chessboard, timeout=1):
        """
        Gets the next best move from the engine.

        Args:
            chessboard (ChessBoard): The chessboard object.
            timeout (int): Timeout in seconds to wait for the best move.

        Returns:
            tuple or None: The best move as (cur_row, cur_col, nxt_row, nxt_col), or None if timed out.
        """
        fen = self.parse_chessboard_to_fenstr(chessboard)
        fen = f"{fen} {self.team}"

        with self.lock:
            self.output.clear()
        self.bestmove = None
        self.bestmove_event.clear()

        self.send_command(f"position fen {fen}")
        self.send_command(f"go depth {self.max_depth}")

        if self.bestmove_event.wait(timeout):
            next_step = self.ucci_move_to_coords(self.bestmove)
            return next_step
        else:
            return None

    def close(self):
        """
        Closes the engine process and terminates the thread.
        """
        self.running = False
        self.send_command("quit")
        self.process.terminate()
