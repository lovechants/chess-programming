import chess.pgn 
import chess
import numpy as np 
def read_pgn(file):
    data = []
    with open(file) as pgn:
        while True:
            game = chess.pgn.read_game(pgn)
            if game is None:
                break
            result = game.headers['Result']
            board = game.board()
            for move in game.mainline_moves():
                board.push(move)
                data.append((board.fen(), result))
    return data 

# Convert Fen into Tensors
def fen_to_tensor(fen):
    board = chess.Board(fen)
    tensor = np.zeros((8,8,17), dtype=np.uint8)

    piece_index ={
        'p': 0, 'n': 1, 'b': 2, 'r': 3, 'q': 4, 'k': 5,
        'P': 6, 'N': 7, 'B': 8, 'R': 9, 'Q': 10, 'K': 11
    }

    for i in range(64):
        piece = board.piece_at(i)
        if piece:
            row = i // 8 
            col = i % 8
            color = 0 if piece.color == chess.WHITE else 6 
            piece_type = piece_index[piece.symbol()]
            tensor[row, col, color + piece_type] = 1
            #print(f'Place {piece.symbol()} at {row}, {col}')
    #print("Tensor slice for black rooks:", tensor[:,:,9])
    #print("Tensor slice for white rooks:", tensor[:,:,9])
    # Castling
    tensor[:, :, 12] = board.has_kingside_castling_rights(chess.WHITE)
    tensor[:, :, 13] = board.has_queenside_castling_rights(chess.WHITE)
    tensor[:, :, 14] = board.has_kingside_castling_rights(chess.BLACK)
    tensor[:, :, 15] = board.has_queenside_castling_rights(chess.BLACK)
    # Turn 
    if board.turn == chess.WHITE:
        tensor[:,:,16] = 1
    else:
        tensor[:,:,16] = 0
    
    return tensor


# Results to a vector
def result_to_vector(result):
    if result == '1-0':
        return 1 # White wins
    elif result == '0-1':
        return -1 # Black wins
    elif result == '1/2-1/2':
        return 0 # Draw
    return 0 # Handle later

"""
data = read_pgn('games.pgn')

tensors = []
labels = []

for fen, result in data:
    tensor = fen_to_tensor(fen)
    label = result_to_vector(result)
    tensors.append(tensor)
    labels.append(labels)

tensors = np.array(tensors)
labels = np.array(labels)
"""
