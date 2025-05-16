from ChessBoard import *
from ChessAI1 import ChessAI
import pygame

pygame.init()
screen = pygame.display.set_mode((1000, 730))
chessboard = ChessBoard(screen)
chessboard.create_chess()
chessboard.printstr()