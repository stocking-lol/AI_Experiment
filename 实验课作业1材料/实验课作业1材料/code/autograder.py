import random
from Game import *
from Dot import *
from ChessBoard import *

import argparse
import os

# Set SDL video driver to "dummy" to avoid opening a GUI window
os.environ["SDL_VIDEODRIVER"] = "dummy"

from ChessAI1 import ChessAI
from ChessAI1 import ChessAI as GradAI

from pikafish_engine import PikafishEngine

from tqdm import tqdm


def play_one_round(ai_grad, ai_test, chessboard, game, max_steps=150):
    """
    Simulates one round of the game between two AI players.

    Args:
        ai_grad: The AI representing the graduate team.
        ai_test: The AI being tested.
        chessboard: The chessboard object.
        game: The game object.
        max_steps: Maximum number of steps allowed before declaring a tie.

    Returns:
        The winning team or "t" for a tie.
    """
    game.reset_game()
    steps = 0
    while True:
        if game.get_player() == ai_grad.team:
            # Graduate AI makes a move
            cur_row, cur_col, nxt_row, nxt_col = ai_grad.get_next_step(chessboard)
            chessboard.move_chess_silent(cur_row, cur_col, nxt_row, nxt_col)
            #chessboard.printstr()
            if chessboard.judge_win(game.get_player()):
                game.set_win(game.get_player())
                break

            game.exchange()
        else:
            # Test AI makes a move
            cur_row, cur_col, nxt_row, nxt_col = ai_test.get_next_step(chessboard)
            chessboard.move_chess_silent(cur_row, cur_col, nxt_row, nxt_col)
            #chessboard.printstr()
            if chessboard.judge_win(game.get_player()):
                game.set_win(game.get_player())
                break

            game.exchange()
        steps += 1
        if steps > max_steps:
            return "t"  # Tie

    return game.get_player()


def main():
    """
    Main function to run the autograder. It parses command-line arguments,
    initializes the game environment, and runs tests for different AI algorithms.
    """
    # Command-line arguments
    parser = argparse.ArgumentParser(description="Chinese Chess")
    parser.add_argument(
        "-t",
        "--test_cases",
        nargs="+",
        type=str,
        help="test cases: [minmax, alphabeta, pikafish]",
    )
    parser.add_argument(
        "--pika_rounds",
        type=int,
        default=100,
        help="how many rounds to play with pikafish",
    )
    # Control Pikafish's skill level
    parser.add_argument(
        "--pika_skill", type=int, default=0, help="skill level of pikafish"
    )
    parser.add_argument(
        "--pika_depth", type=int, default=1, help="max depth of pikafish search"
    )
    parser.add_argument(
        "--pika_multipv", type=int, default=100, help="principle variations of pikafish engine, higher leads to lower performance"
    )
    args = parser.parse_args()

    # Initialize pygame
    pygame.init()
    screen = pygame.display.set_mode((1000, 730))

    # Create chessboard object
    chessboard = ChessBoard(screen)
    # Create game object (encapsulates current player, game state, etc.)
    game = Game(screen, chessboard)
    game.back_button.add_history(chessboard.get_chessboard_str_map())

    # Test 1: Test the minmax algorithm
    if "minmax" in args.test_cases:
        print("Testing minmax algorithm...")
        ai_grad = GradAI(game.computer_team, max_depth=1, search_with="alphabeta")
        ai_test = ChessAI(game.user_team, max_depth=3, search_with="minmax")

        try:
            result = play_one_round(ai_grad, ai_test, chessboard, game)
        except Exception as e:
            print(f"Error: {e}")
            import traceback;
            traceback.print_exc()
            result = None

        if result == game.user_team:
            print(f"Winner team: {result}. Test minmax pass")
        else:
            print(f"Winner team: {result}. Test minmax fail")

    # Test 2: Test the alphabeta algorithm
    if "alphabeta" in args.test_cases:
        print("Testing alphabeta algorithm...")
        ai_grad = GradAI(game.computer_team, max_depth=1, search_with="alphabeta")
        ai_test = ChessAI(game.user_team, max_depth=4, search_with="alphabeta")

        try:
            result = play_one_round(ai_grad, ai_test, chessboard, game)
        except Exception as e:
            print(f"Error: {e}")
            result = None

        if result == game.user_team:
            print(f"Winner team: {result}. Test alphabeta pass")
        else:
            print(f"Winner team: {result}. Test alphabeta fail")

    # Test 3 [optional]: Play against the neural network engine Pikafish
    if "pikafish" in args.test_cases:
        ai_grad = PikafishEngine(
            game.computer_team,
            max_depth=args.pika_depth,
            skill_level=args.pika_skill,
            multi_pv=args.pika_multipv,
        )
        ai_test = ChessAI(game.user_team, max_depth=3, search_with="alphabeta")

        results = [0, 0, 0]  # [win, lose, tie]
        pbar = tqdm(total=args.pika_rounds, unit="round")
        for i in range(args.pika_rounds):
            ai_test.reset()
            winteam = play_one_round(ai_grad, ai_test, chessboard, game)
            if winteam == game.user_team:
                results[0] += 1
            elif winteam == game.computer_team:
                results[1] += 1
            else:
                results[2] += 1
            pbar.set_description(
                f"Pikafish vs Yours, round {i + 1}/{args.pika_rounds}. Win: {results[0]}, Lose: {results[1]}, Tie: {results[2]}"
            )
            pbar.update(1)

        ai_grad.close()


if __name__ == "__main__":
    main()
