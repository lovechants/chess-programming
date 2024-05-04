import chess
import tensorflow as tf 
from pgn_processing import result_to_vector
from game_eval import * 
import datetime
'''
def calculate_reward(board):
    if board.is_game_over():
        outcome = board.outcome()
        if outcome.winner is None:
            result = 0 
            return result
        elif outcome.winner:
            if board.turn == chess.WHITE:
        result = 1
            else: 
                result = -1
        else:
            if board.turn == chess.WHITE:
                result = -1
            else:
                result = 1
    else:
        return heuristic_eval(board) 


def process(data, model):
    input_tensors = []
    move_indicies = []
    reward_targets = []
    for fen, best_move, reward in data:
        board = chess.Board(fen)
        tensor = fen_to_tensor(fen)
        input_tensors.append(tensor)
        legal_moves = list(board.legal_moves)
        try:
            move_index = legal_moves.index(best_move)
            move_indicies.append(move_index)
        except ValueError:
            print("Move not legal for board state")
            continue 
        reward_targets.append(reward)
    if not input_tensors:
        print("No valid data to train on")
        return 

    input_tensors = tf.stack(input_tensors)
    targets = tf.one_hot(move_indicies, depth=64)
    reward_targets = tf.convert_to_tensor(reward_targets, dtype=tf.float32)
    print("Shapes:", input_tensors.shape, targets.shape, reward_targets.shape)
    train(input_tensors, targets, reward_targets, model)

def train(input_tensors, targets, rewards, model, learning_rate=0.001):
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    with tf.GradientTape() as tape:
        policy_outputs, value_outputs = model(input_tensors, training=True)
        policy_loss = tf.keras.losses.CategoricalCrossentropy()(targets, policy_outputs)
        value_loss = tf.keras.losses.MeanSquaredError()(rewards, value_outputs)
        total_loss = policy_loss - value_loss

    gradients = tape.gradient(total_loss, model.training_variables)
    optimizer.apply_gradients(zip(gradients, model.training_variables))
'''

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

def train_network(model, inputs, policy_targets, value_targets, epochs=20, batch_size=32):
    #print("Inputs shape:", inputs.shape)
    #print("Policy targets shape:", policy_targets.shape)
    #print("Value targets shape:", value_targets.shape)
    log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    history = model.fit(inputs, [policy_targets, value_targets], epochs=epochs, batch_size=batch_size, verbose=1, callbacks=[tensorboard_callback])
    return history



