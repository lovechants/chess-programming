import chess
import numpy as np 
from pgn_processing import fen_to_tensor
from neural_net import * 
# For MCTS eval dependant on needs 

def simple_eval(board):
    outcome = board.outcome()
    if outcome is None: # Draw
        return 0
    if outcome.winner is None: # Draw
        return 0
    elif outcome.winner:
        return 1 if board.turn == chess.WHITE else -1 # + White win, - Black win 
    else:
        return -1 if board.turn == chess.WHITE else 1  # - White loss, + Black loss

def heuristic_eval(board):  # Eval at material values at position  
    values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
    white_score = sum(values[piece.piece_type] for piece in board.piece_map().values() if piece.color == chess.WHITE)
    black_score = sum(values[piece.piece_type] for piece in board.piece_map().values() if piece.color == chess.BLACK)
    score = white_score - black_score
    return score if board.turn == chess.WHITE else -score


def mobility_eval(board):
    mobility = 0
    for move in board.legal_moves:
        if board.piece_at(move.from_square).color == chess.WHITE:
            mobility += 1
        else:
            mobility -= 1
    return mobility


def king_safety(board):
    king_position = board.king(chess.WHITE)
    safety = 0
    # Example: Check if the squares around the king are protected
    for offset in [-9, -8, -7, -1, 1, 7, 8, 9]:
        square = king_position + offset
        if square >= 0 and square < 64 and board.is_attacked_by(chess.BLACK, square):
            safety -= 1
    return safety

def combinded_eval(board):
    material = heuristic_eval(board)
    mobility = mobility_eval(board) * .1
    king_pos = king_safety(board) * .2
    total_score = material + mobility + king_pos
    return total_score
