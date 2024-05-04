from model import convert_board, encode_move
import chess.pgn

def read_pgn(file):
    features = []
    targets = []
    with open(file) as pgn:
        while True:
            game = chess.pgn.read_game(pgn)
            if game is None:
                break
            board = game.board()
            for move in game.mainline_moves():
                board.push(move)
                tensor = convert_board(board)
                encoded_move = encode_move(move)
                features.append(tensor)
                targets.append(encoded_move)
    return features, targets
#data = read_pgn('20_games.pgn')

