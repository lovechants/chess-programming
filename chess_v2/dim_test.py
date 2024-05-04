from lib import * 

def test_expand_function():
    # Set up a simple board state
    board = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    model = create_model(8, 256, 5)  # Assuming create_model is your function to create the neural network
    node = Node(state=board.fen(), model=model, board=board)

    # Mock policy output, pretend we have a perfect policy where index 0 (move 'g1f3') is the best move
    mock_policy = np.zeros((1, 4672))
    move_map = initialize_move_index()
    best_move_uci = 'g1f3'
    best_move_index = move_map[best_move_uci]
    mock_policy[0, best_move_index] = 1  # Set highest score to the best move

    # Force the model to return the mock policy
    model.predict = lambda x: (mock_policy, None)

    # Run expand
    expand(node)

    # Check outputs
    if node.children:
        print("Expansion successful, children added:")
        for child in node.children:
            print("Child move:", child.move.uci())
    else:
        print("No children added, check move processing and policy application.")

def test_expand_function_debug():
    board = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    model = create_model(8, 256, 5)
    node = Node(state=board.fen(), model=model, board=board)

    move_map = initialize_move_index()
    print("Move Map Size:", len(move_map))

    # Mock a policy output for debugging
    mock_policy = np.random.rand(1, 4672)  # Using random values for demonstration
    model.predict = lambda x: (mock_policy, None)

    # Run expand with debugging prints
    expand(node, debug=True)  # Assuming 'expand' function accepts a 'debug' parameter to enable verbose output

test_expand_function()
test_expand_function_debug()


