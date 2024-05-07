from lib import * 
from keras.callbacks import LambdaCallback, ModelCheckpoint
import csv 
"""
Typical Args
board_size = 8 (8x8)
num_channels = 256
Where n_layers can be increased for a deeper network

"""

def create_model(board_size, num_channels, n_layers):
    shape = (board_size, board_size, 17) # 17 = Standard chess(12) + castling and promotion channels
    inputs = Input(shape=shape)
    
    # Convolution
    x = Conv2D(num_channels, kernel_size=3, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    


    for _ in range(n_layers):
        x = Conv2D(num_channels, kernel_size=3, padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
    #print(f"Addition layers, {n_layers} Successfully created")
    
    # Policy
    policy_conv = Conv2D(2, kernel_size=1)(x)
    policy_flat = Flatten()(policy_conv)
    policy_output = Dense(4096, activation='softmax', name='policy_output')(policy_flat)
    #print("Policy layers and output created")

    # Value 
    value_conv = Conv2D(1, kernel_size=1)(x)
    value_flat = Flatten()(value_conv)
    value_hidden = Dense(64, activation='relu')(value_flat)
    value_output = Dense(1, activation='tanh', name='value_output')(value_hidden)

    #print("Value layers and output created")
    model = Model(inputs=inputs, outputs=[policy_output, value_output])
    model.compile(optimizer='adam', loss={'policy_output':'categorical_crossentropy', 'value_output':'mean_squared_error'}, metrics={'policy_output': 'accuracy', 'value_output': 'mse'})
    
    #print("Model Created and Compiled")
    return model 


def initialize_move_index():
    move_index = {}
    columns = 'abcdefgh'
    rows = '12345678'
    index = 0 
    for start_col in columns:
        for start_row in rows:
            for end_col in columns:
                for end_row in rows:
                    move = f"{start_col}{start_row}{end_col}{end_row}"
                    move_index[move] = index
                    index+=1 
    return move_index 

def convert_move_to_index(move_uci, move_map):
    return move_map.get(move_uci, -1)

def process_game_data(game_data, win_score, move_map):
    training_data = []
    for state, move_uci, _ in game_data:
        move_index = convert_move_to_index(move_uci, move_map)
        if move_index != -1:
            training_data.append((state, move_index, win_score))
    return training_data

def process_training_data(data, num_moves=4096):
    inputs = np.array([fen_to_tensor(state) for state, _, _ in data])
    policy_indices = [policy for _, policy, _ in data]
    policy_targets = np.zeros((len(data), num_moves))
    for i, idx in enumerate(policy_indices):
        if idx >= 0:
            policy_targets[i, idx] = 1 

    value_targets = np.array([value for _, _, value in data]).reshape(-1,1)
    
    return inputs, policy_targets,value_targets

checkpoint_path = "best_model.h5"
#checkpoint_callback=ModelCheckpoint(
#        checkpoint_path,
#        monitor='val_loss',
#        save_best_only=True,
#        verbose=1,
#        mode='min')

checkpoint = ModelCheckpoint(
        checkpoint_path,
        monitor='val_loss',
        save_best_only=True,
        verbose=1,
        mode='min')

def log_training(epoch, logs):
    with open ('data_logs/training_metrics.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([epoch, logs.get('policy_output_loss'), logs.get('value_output_loss')])

def train_network(model, inputs, policy_targets, value_targets, epochs=20, batch_size=32): 
    #log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    #tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    history = model.fit(
            inputs, 
            [policy_targets, value_targets], 
            epochs=epochs, 
            batch_size=batch_size, 
            verbose=1,
            callbacks=[checkpoint, LambdaCallback(on_epoch_end=lambda epoch, logs: log_training(epoch,logs))]
    )

    #, callbacks=[tensorboard_callback])
    return history
    
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

def result_to_vector(result):
    if result == '1-0':
        return 1 # White wins
    elif result == '0-1':
        return -1 # Black wins
    elif result == '1/2-1/2':
        return 0 # Draw
    return 0 # Handle later
