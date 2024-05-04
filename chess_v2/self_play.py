from mcts import * 
"""
Self play environment 
"""
"""
EXAMPLE NODE FOR MCTS
print("Testing with a custom position")
custom_fen = "r1bqkbnr/pppp1ppp/2n5/2p1p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"  # An example from the Sicilian Defense
custom_board = chess.Board(custom_fen)
custom_state = custom_board.fen()

test_root2 = Node(custom_state, model) # Create the root node with the custom starting state
mcts(test_root2, 20)


def self_play(node, num_games, depth):
    for _ in range(num_games):
        while not node.board.is_game_over():
            node.move, _ = mcts(node, depth)
            print(f"Move Made: {node.move} | Board State:\n{node.board}")
        print(f"Final Board State:\n{node.board}")
        print(f"Results: {node.board.result()}")
        node.board.reset()


            
def self_simulate(node, model, max_steps, move_map):
    board = node.board.copy()
    data = []
    training_data = []
    print("Starting simulation with board state:", board.fen())
    steps = 0
    with open(f'games_fen_iteration_{steps+1}.txt', 'w') as file:
        while not board.is_game_over() and steps < max_steps:
                policy, _ = model.predict(fen_to_tensor(board.fen())[np.newaxis, :])
                print(f"Policy output: {policy}")
                best_move = predict_move(board, policy)
                print(f"Simulating move {best_move} at step {steps}")
                board.push(best_move)
                print(f"Simulated move: {best_move}, Resulting board:\n {board}")
                print(f"Turn after move {node.move}: {'White' if node.board.turn == chess.WHITE else 'Black'}")
                #time.sleep(.1)
                data.append((board.fen(), best_move.uci(), policy))
                file.write(str(board.fen())+'\n')
                steps += 1
                print("Updated board state:", board.fen())
                node.move = best_move
        
    #board.push(best_move)
    #node.move = best_move
    node.board = board 
    result = combinded_eval(board)
    game_data = process_game_data(data, result, move_map)
    training_data.extend(game_data)
    inputs, policy_targets, value_targets = process_training_data(game_data)
    print(f"Inputs shape: {inputs.shape}, Policy Target Shapes: {policy_targets.shape}, Value Target Shapes: {value_targets.shape}")
    if inputs.size == 0 or policy_targets.size == 0 or value_targets.size == 0:
        print("Training data is empty or misaligned")
    
    print("Training model...")
    train_network(model, inputs, policy_targets, value_targets)

    # Save the model after each training iteration
    model_path = f'model_iteration_{steps + 1}.h5'
    model.save(model_path)
    print(f"Model saved as {model_path}")

"""
