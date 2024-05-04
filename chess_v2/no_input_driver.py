from lib import *
"""
def self_play_and_training(model, num_games, iterations, depth, move_map): # num_games really is = to num moves
    for iteration in range(iterations):
        print(f"Starting training interation {iteration + 1}")
        game_data = []
        for game_num in range(num_games):
            starting_pos = chess.STARTING_FEN
            board = chess.Board(starting_pos)
            root = Node(starting_pos, model, board=board)
            fens = []

            while not board.is_game_over():
                node, data_per_game = mcts(root, depth)
                fens.append(node.state)
                root = node
                print(f"Move Made: {node.move} | Board State:\n{node.board}")
                if node.board.is_game_over():
                    print("Game Over detected", board.result())
                    break

                with open(f'game_fen_{game_num + 1}_iteration_{iteration+1}.txt', 'w') as file:
                    for fen in fens:
                            file.write(fen +'\n')
            print(f"game {game_num+1}: Moves Logged")
            print(f"Final board:\n{root.board}\n Winner: {board.result()}")
            board.reset()
            print("Resetting Board for next game")
    
            game_data.extend(data_per_game)
        inputs, policy_targets, value_targets = process_training_data(game_data)
        
        print("Training Model . . .")
        train_network(model, inputs, policy_targets, value_targets,)
        
        model.save(f'model_iteraion_{iteration+1}.h5')
        print("Model Saved")
        
        
        TODO finish creating a test set, need like 40 or so more games for it to test on
        if (iteration + 1) % 10 == 0:
            print(f"Evaluating model at iteration: {iteration + 1}")
            evaluate_model(model, game_data)
        


def self_play_and_training(model, num_games, iterations, depth, move_map):
    for iteration in range(iterations):
        print(f"Starting training iteration {iteration + 1}")
        game_data = []

        for game_num in range(num_games):
            #fens = []
            board = chess.Board()  # Ensures starting from the initial position
            root = Node(board.fen(), model, board=board)
            print(f"\nGame {game_num + 1} started.")
            unchanged = 0
            prev_state = None
            with open(                file.write('Beginning of games\n')
                while not board.is_game_over():
                    if prev_state == board.fen():
                        unchanged += 1
                    else:
                        unchanged = 0

                    if unchanged >= 3:
                        print("No change in state, breaking")
                        break
                    
                    prev_state = board.fen()
                    file.write(prev_state+'\n')
                    node, data_per_game = mcts(root, depth)
                    if node.move in board.legal_moves:
                        board.push(node.move)
                        print(f"move made: {node.move}")

                        # After pushing a move, check the turn to ensure it has switched
                        print(f"After move: {node.move}, it is now {'white' if board.turn == chess.WHITE else 'black'}'s turn.")
                    else:
                        print("No move or illegal move returned from MCTS")
                        continue

                    node.parent = root
                    root = node  # Continue from the new node
                    #board = root.board.copy()
                    #fens.append(node.state)
                    game_data.extend(data_per_game)
                
                    # Debug outputs
                    print(f"Move Made: {node.move}")
                    print(f"Current Board State:\n{board}\n")

                   
                    if board.is_game_over():
                        print("Game Over detected. Reason:", board.result())
                        break
                file.write('End of Game\n')
            # Log the game's final state and reset for the next game
            print(f"Final board state:\n{board}")
            print(f"Game {game_num + 1} complete. Result: {board.result()}\n")
            board.reset()

        # Process collected data and train the model
        inputs, policy_targets, value_targets = process_training_data(game_data)
        print(f"Inputs shape: {inputs.shape}, Policy Target Shapes: {policy_targets.shape}, Value Target Shapes: {value_targets.shape}")
        if inputs.size == 0 or policy_targets.size == 0 or value_targets.size == 0:
            print("Training data is empty or misaligned")
            continue

        print("Training model...")
        train_network(model, inputs, policy_targets, value_targets)

        # Save the model after each training iteration
        model_path = f'model_iteration_{iteration + 1}.h5'
        model.save(model_path)
        print(f"Model saved as {model_path}")
tf.config.run_functions_eagerly(True)
 
def evaluate_model(model, test_data):
    inputs, policy_targets, value_targets = process_training_data(test_data)
    loss, policy_loss, value_loss = model.evaluate(inputs, [policy_targets, value_targets], verbose=0)
    print(f"Evaluation loss: {loss}, Policy Loss: {policy_loss}, Value Loss: {value_loss}")



def self_play_and_training(model, num_games, iterations, depth, move_map):
    for iteration in range(iterations):
        print(f"Starting training iteration {iteration + 1}")
        game_data = []

        for game_num in range(num_games):
            board = chess.Board()
            root = Node(board.fen(), model, board=board)
            print(f"\nGame {game_num + 1} started.")
            while not board.is_game_over():
                node, <D-u>data_per_game = mcts(root, depth)
                if node.move in board.legal_moves:
                    board.push(node.move)
                    print(f"Move made: {node.move}, Board state: {board.fen()}")
                else:
                    print(f"Illegal move attempted: {node.move.uci()}")
                    continue  # Or handle differently

                # Update root to the new node, reflecting the latest game state
                root = node
                game_data.extend(data_per_game)

                if board.is_game_over():
                    print("Game Over detected. Reason:", board.result())
                    break

            print(f"Final board state:\n{board}")
            print(f"Game {game_num + 1} complete. Result: {board.result()}\n")
            board.reset()

        # Process and train
        inputs, policy_targets, value_targets = process_training_data(game_data)
        train_network(model, inputs, policy_targets, value_targets)
"""

def self_simulate(node, model, max_steps, move_map):
    board = node.board.copy()
    data = []
    training_data = []
    policy = None 
    print("Starting simulation with board state:", board.fen())
    steps = 0
    with open(f'Game_fens_trial_3.txt', 'w') as file:
        while not board.is_game_over() and steps < max_steps:
                policy, _ = model.predict(fen_to_tensor(board.fen())[np.newaxis, :])
                #print(f"Policy output: {policy}")
                best_move = predict_move(board, policy)
                print(f"Simulating move {best_move} at step {steps}")
                board.push(best_move)
                print(f"Simulated move: {best_move}, Resulting board:\n {board}")
                print(f"Turn after move {node.move}: {'White' if node.board.turn == chess.WHITE else 'Black'}")
                #time.sleep(.1)
                file.write(board.fen()+'\n')
                steps += 1
                print("Updated board state:", board.fen())
                node.move = best_move
                policy = policy
                if board.is_game_over():
                    print("Game Over detected. Reason:", board.result())

    data.append((board.fen(), node.move.uci(), policy))
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
    model_path = f'model_iteration_tensorboard3.h5'
    model.save(model_path)
    print(f"Model saved as {model_path}")
    return model, node
tf.config.run_functions_eagerly(True)




def main():
    model = create_model(8,256,64)
    #print(model.summary())
    move_index_map = initialize_move_index()
    opening_pos = chess.STARTING_FEN
    opening_tensor = fen_to_tensor(opening_pos)
    tensor = np.expand_dims(opening_tensor, axis=0)
    output = model.predict(tensor)
    policy_output, value_output = output
    #self_play_and_training(model, 5, 2, 1, move_index_map)
    node = Node(chess.STARTING_FEN, model)
    for _ in range(30):
        model,node = self_simulate(node, model, 70, move_index_map)
        reset = chess.Board(opening_pos)
        node.board = reset

        
    # Print the shape of each part of the output
    #board = chess.Board(chess.STARTING_FEN)  # Ensures starting from the initial position
    #root = Node(board.fen(), model, board=board)
    #self_play(root, 10,4)
    #print(model.summary())
    #policy_output_shape = model.output_shape[0][1]
    #print("Policy Output Shape:", policy_output_shape)
    #print("Move Map Size:", len(move_index_map))

    # Example test
    #test_board = chess.Board()
    #test_move = chess.Move.from_uci('e2e4')
    #move_map = initialize_move_index()
    #test_index = convert_move_to_index(test_move.uci(), move_map)
    #print("Test Move Index:", test_index)
if __name__ == "__main__":
    main()





