from pgn_processing import * 
from neural_net import * 
from mcts import * 
from game_eval import * 
from movement import * 
from self_play import self_play
"""
# Sample FEN string
fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"  # Starting position
tensor = fen_to_tensor(fen)
'''
# Check specific pieces' positions
assert tensor[7, 0, 9] == 1, "Rook is missing at (7, 0)"  # Black Rook at h8
assert tensor[7, 7, 9] == 1, "Rook is missing at (7, 7)"  # White Rook at h1
assert tensor[0,7, 3] == 0, "Random Test"
'''


#assert tensor[0, 7, 5] == 1, "Rook is missing at (0, 7)"  # White Rook at h1
# Check castling rights (should be present for both kingside and queenside)
assert tensor[0, 0, 12] == 1, "White kingside castling right missing"
assert tensor[0, 0, 13] == 1, "White queenside castling right missing"
assert tensor[0, 0, 14] == 1, "Black kingside castling right missing"
assert tensor[0, 0, 15] == 1, "Black queenside castling right missing"

# Check turn indicator (should be white's turn)
assert tensor[0, 0, 16] == 1, "Turn indicator incorrect, should be White's turn"
print("All tests passed. Tensor is correctly configured.")

"""



print("Testing Neural Network Implementation")
# Example FENs 
fen_samples = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",  # Starting position
    "r1bqkbnr/pppp1ppp/2n5/2p1p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",  # Open Sicilian
    "6k1/5ppp/8/8/8/8/5PPP/6K1 w - - 0 1",# King and pawns endgame
    "r1bqk1nr/ppp3pp/5p2/2bPp3/3N4/2N5/PPP2PPP/R1BQKB1R w KQkq - 1 7", # Random Me and Nelson game
    "6r1/pp1k1p1p/2p2pPB/8/6P1/7P/PPP5/4R1K1 b - - 0 22"
]

model = create_model(8,256,2)
# Convert FENs to tensors
tensors = np.array([fen_to_tensor(fen) for fen in fen_samples])

# Assuming 'model' is your loaded and compiled neural network
predictions = model.predict(tensors)

# Display the predictions
'''
def predict_move(fen, policy):
    board = chess.Board(fen) 
    legal_moves = list(board.legal_moves)
    if len(legal_moves) != len(policy):
        print("Number of legal moves and policy output size do not match")
        policy_output = masking(policy, legal_moves, board)
    
    best_move_index = np.argmax(policy_output)
    best_move = legal_moves[best_move_index]
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
'''
"""
for i, (policy, values) in enumerate(zip(predictions[0], predictions[1])):
    predicted_move = predict_move(chess.Board(fen_samples[i]), policy)
    print(f"Predictions for FEN {i + 1}: {fen_samples[i]}")
    print("Predicted Move:", predicted_move)
    print("Policy (probabilities of moves):", policy)
    print("Value (estimated outcome):", values)
"""

"""
print("Testing MCTS")
#print("Testing with opening position")
#test_fen = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'
#test_board = chess.Board(test_fen)
#initial_state = test_board.fen()
#test_root1 = Node(initial_state, model)
#mcts(test_root1, 1)

print("Testing with a custom position")
#custom_fen = "r1bqkbnr/pppp1ppp/2n5/2p1p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"  # An example from the Sicilian Defense
custom_fen = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'
custom_board = chess.Board(custom_fen)
custom_state = custom_board.fen()
print("Initial Board Positions:")
print(custom_board)
test_root2 = Node(custom_state, model, board=custom_board) # Create the root node with the custom starting state
# r1bqkbnr/pppp1ppp/2n5/2p1p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3
test_root2 = mcts(test_root2, 1)
print(test_root2.board) 

"""
print("testing self play")

initial_fen = chess.STARTING_FEN
root = Node(initial_fen, model, board=chess.Board(initial_fen))
self_play(root,2,1)
