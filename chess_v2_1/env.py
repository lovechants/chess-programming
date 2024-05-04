from lib import * 
from nnue import * 

def self_simulate(node, model, max_steps, move_map):
    board = node.board.copy()
    data = []
    training_data = []
    policy = None 
    print("Starting simulation with board state:", board.fen())
    steps = 0
    with open(f'fens_from_game', 'w') as file:
        while not board.is_game_over() and steps < max_steps:
                policy, _ = model.predict(fen_to_tensor(board.fen())[np.newaxis, :])
                best_move = predict_move(board, policy)
                print(f"Simulating move {best_move} at step {steps}")
                board.push(best_move)
                print(f"Simulated move: {best_move}, Resulting board:\n {board}")
                time.sleep(.1)
                file.write(board.fen()+'\n')
                steps += 1
                print("Updated board state:", board.fen())
                time.sleep(.2)
                node.move = best_move
                policy = policy
                if board.is_game_over():
                    print("Game Over detected. Reason:", board.result())

    data.append((board.fen(), node.move.uci(), policy))
    node.board = board 
    result = combinded_eval(board)
    game_data = process_game_data(data, result, move_map)
    training_data.extend(game_data)
    inputs, policy_targets, value_targets = process_training_data(game_data)
    if inputs.size == 0 or policy_targets.size == 0 or value_targets.size == 0:
        print("Training data is empty or misaligned")
    
    print("Training model...")
    time.sleep(.1)
    train_network(model, inputs, policy_targets, value_targets)

    # Save the model after each training iteration
    #model_path = f'model_iteration_tensorboard3.h5'
    #model.save(model_path)
    #print(f"Model saved as {model_path}")
    return model, node

# This function has taken up so many hours atp
def predict_move(board, policy):
    if not isinstance(board, chess.Board):
        raise ValueError("Board must be a chess.Board object")
    
    legal_moves = list(board.legal_moves)
    move_indices = {move.uci(): i for i, move in enumerate(legal_moves)}
    policy = add_noise(policy)
    
    # Ensure policy is reshaped correctly if necessary; assuming policy shape from model is (1, 4672)
    if policy.ndim == 1:
        policy = policy.reshape(1, -1)

    # Initialize the mask with the correct dimensions
    mask = np.zeros_like(policy, dtype=bool)

    # Set valid entries in mask
    valid_moves=False
    for move in legal_moves:
        move_uci = move.uci()
        index = move_indices.get(move_uci, -1)
        if index >= 0 and index < policy.shape[1]:
            mask[0,index] = True
            valid_moves = True
            print(f"Move: {move_uci}, Index: {index}, Policy Score: {policy[0][index]}")
        else:
            print(f"Move: {move_uci}, Index: {index} - Index out of bounds")
    if not valid_moves:
        print("No valid moves found")
        return None 
    # Apply mask to policy and select the best move
    masked_policy = np.where(mask, policy, 0)
    best_move_idx = np.argmax(masked_policy)
    if masked_policy[0, best_move_idx] == 0:
        print(" No valid moves have positive scores")
        pass

    best_move = legal_moves[best_move_idx % len(legal_moves)]  # Modulo operation to ensure index is within bounds

    return best_move

def add_noise(prob, noise=0.39):
    noise = np.random.normal(0, noise, prob.shape)
    noisy_prob = prob + noise 
    noisy_prob = np.clip(noisy_prob, 0,1)
    noisy_prob /= np.sum(noisy_prob, axis=-1, keepdims=True)
    return noisy_prob

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
