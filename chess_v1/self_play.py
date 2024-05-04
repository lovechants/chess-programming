from model import *
import numpy as np 
# for later engine eval 
# import chess.engine 
"""
engine = "path to engine"
def eval_pos(board):
    engine = chess.engine.SimpleEngine.popen_uci(engine)
    info = engine.analyse(board, chess.engine.Limit(time=.1))
    engine.quit()
    return info["score"]
"""

def predict_move(model, board):
    input_t = convert_board(board)
    input_t = np.expand_dims(input_t, axis=0)
    policy_output, _ = model.predict(input_t)
    move_index = np.argmax(policy_output)
    move = index_move(move_index, board)
    return move

def index_move(index, board):
    legal_moves = list(board.legal_moves)
    return legal_moves[index % len(legal_moves)]

def play_self(prev_model, model, n_games): 
    games = []
    outcomes = {'win': 0, 'loss': 0, 'draw':0}
    for _ in range(n_games):
        board = chess.Board()
        if np.random.choice([True, False]):
            white_model = prev_model
            black_model = model
        else:
            white_model = model
            black_model = prev_model
        while not board.is_game_over():
            if board.turn == chess.WHITE:
                move = predict_move(white_model, board)
            else:
                inverted_board = board.mirror()
                move = predict_move(black_model, inverted_board)
                move = chess.Move(from_square=chess.square_mirror(move.from_square),
                                  to_square=chess.square_mirror(move.to_square))
            board.push(move)
            encoded_move = encode_move(move)
            games.append({'fen':board.fen(), 'move': encoded_move})
        result = board.result()
        if result == '1-0':
            outcomes['win'] += 1 # assuming white wins
        elif result == '0-1':
            outcomes['loss'] += 1 # assuming black wins
        elif result == '1/2-1/2':
            outcomes['draw'] += 1
        board.reset()
    return games, outcomes


