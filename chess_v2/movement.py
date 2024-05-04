import chess
from neural_net import *
import numpy as np
from exploratory import add_noise
"""
def predict_move(board, policy):

    if not isinstance(board, chess.Board):
        raise ValueError("The board must be a chess.Board object.")
    legal_moves = list(board.legal_moves)
    if len(legal_moves) != len(policy):
        print("Number of legal moves and policy output size do not match. Normalizing lengths.")
        #policy_output = masking(policy, legal_moves, board)
        policy_output = policy[:len(legal_moves)]
    else: 
        policy_output = policy
    
    best_move_index = np.argmax(policy_output)
    if best_move_index >= len(legal_moves):
        best_move_index = best_move_index % len(legal_moves)
    
    best_move = legal_moves[best_move_index]
    print(f"Best move selected: {best_move}")
    return best_move

def masking(policy_output, legal_moves, board):
    mask = np.zeros(len(policy_output))
    for i, move in enumerate(board.legal_moves):
        mask[i] = 1 
    masked_policy = policy_output * mask
    if np.sum(masked_policy) > 0:
        normalized_policy = masked_policy / np.sum(masked_policy)
    else:
        normalized_policy = masked_policy
    return normalized_policy
"""
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
