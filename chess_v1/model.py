import chess
import numpy as np 
import tensorflow as tf 
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Input, Activation
import tensorflow.keras.backend as K 
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam # issue with m chip mac replaced with legacy  
from tensorflow.keras.regularizers import l2


# Tensor modeling 
def convert_board(b):
    tensor = np.zeros((8,8,14), dtype=np.float32) # 8 x 8 x 14 board + tensor represnetiation

    piece_index = {
            chess.PAWN: 0,
            chess.KNIGHT: 1, 
            chess.BISHOP: 2,
            chess.ROOK: 3,
            chess.QUEEN: 4,
            chess.KING: 5
            }

    for square in chess.SQUARES:
        piece = b.piece_at(square)
        if piece is not None: 
            row, col = divmod(square, 8)
            index = piece_index[piece.piece_type]
            if piece.color == chess.WHITE:
                tensor[row, col, index] = 1
            else:
                tensor[row,col, index + 6] = 1
    
    # Castling
    if b.has_queenside_castling_rights(chess.WHITE):
        tensor[:, :, 12] = 1
    if b.has_kingside_castling_rights(chess.WHITE):
        tensor[:, :, 12] = 2
    if b.has_queenside_castling_rights(chess.BLACK):
        tensor[:, :, 12] = 3
    if b.has_kingside_castling_rights(chess.BLACK):
        tensor[:, :, 12] = 4

    # Handle en passant square
    if b.ep_square:
        ep_row, ep_col = divmod(b.ep_square, 8)
        tensor[ep_row, ep_col, 13] = 1 

    return tensor

board = chess.Board()
tensor = convert_board(board)

def custom_loss(y_true, y_pred):
    y_true = K.cast(y_pred, y_pred.dtype)
    loss = K.mean(K.square(y_true - y_pred))

    return loss

def create_model(genome):
    try:
        if len(genome['filters']) != genome['num_layers'] or len(genome['activations']) != genome['num_layers']:
            print("Genome mismatched lengths")
            return None 

        input_layer = Input(shape=(8,8,14))
        x = input_layer
        
        for i in range(genome['num_layers']):
            x = Conv2D(genome['filters'][i], (3,3), padding='same')(x)
            x = Activation(genome['activations'][i])(x)
        x = Flatten()(x)

        policy_output = Dense(64, activation='softmax')(x)
        value_output = Dense(1, activation='tanh')(x)

        model = Model(inputs=input_layer, outputs=[policy_output, value_output])
        model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=genome['learning_rate']), loss='mean_squared_error', metrics=['mse'])

        return model
    except Exception as e:
        print(f"Failed to create model: {str(e)}")
        return None

def encode_move(move):
    from_square = move.from_square
    to_square = move.to_square
    promotion = move.promotion if move.promotion is not None else 0
    
    move_encoded = from_square * 1000 + to_square * 10 + promotion
    return move_encoded

def min_max_prune(board, depth, alpha, beta, player, model):
    if depth == 0 or board.is_game_over():
        input_tensor = convert_board(board)
        value = model.predict(np.array([input_tensor]))[1]
        return value 
    if player:
        max_eval = -float("inf")
        for move in board.legal_moves:
            board.push(move)
            eval = min_max_prune(board, depth-1, alpha, beta, False, model)
            board.pop()
            max_eval = max(max_eval,eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = float('inf')
        for move in board.legal_moves:
            board.push(move)
            eval = min_max_prune(board, depth-1, alpha, beta, True, model)
            board.pop()
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval

def eval_network(model, data):
    results = []
    for game in data:
        #print("Current game data:", game)
        if isinstance(game, dict) and 'fen' in game:
            board = chess.Board(game['fen'])
            result = min_max_prune(board, 3, -float('inf'), float('inf'), True, model)
            results.append(result)
        else: 
            print(f"FEN Issue")
    fitness = np.mean(results)
    return fitness # Fitness metric



