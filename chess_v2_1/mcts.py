from lib import * 

class Node:
    def __init__(self, state, model, parent = None, move=None, board=None):
        self.state = state
        if parent is None:
            self.parent = None
        else:
            self.parent = parent
        self.move = move
        self.children = []
        self.visit_count = 0 
        self.win_score = 0
        self.model = model
        if board is None:
            self.board = chess.Board(state)
        else:
           self.board = board.copy()
            

    def add_child(self, child):
        self.children.append(child)

    def update(self, win_score):
        self.visit_count += 1 
        self.win_score += win_score
    def is_leaf(self):
        return not self.children

# Creating the MCTS // v1 was genetic algorithm 

def select_child(node):
    if node.board is None:
        raise Exception("selecting child from a node with no board")
    unvisted = [child for child in node.children if child.visit_count == 0]
    if unvisted:
        selected = random.choice(unvisted)
        selected.visit_count += 1
        return selected

    log_vertex = math.log(node.visit_count)
    exploration_constant = 2.39
    return max(node.children, key=lambda x: x.win_score / x.visit_count + exploration_constant *  math.sqrt(2 * log_vertex / x.visit_count))

def expand(node, debug=False):
    if node.board is None:
        raise Exception("trying to expand a node without a valid board object")
    legal_moves = list(node.board.legal_moves)
    if not legal_moves:
        print("no legal moves available to expand")
        return
    board = node.board.copy()
    print(f"Expanding node for board state: {board.fen()} with {len(legal_moves)} legal moves")
    move_map = initialize_move_index()
    tensor = fen_to_tensor(board.fen())[np.newaxis, :]
    policy, _ = node.model.predict(tensor)
    policy = add_noise(policy)
    if debug:
        print(f"Expanding node for board state: {node.board.fen()} with {len(list(node.board.legal_moves))} legal moves.")

    tensor = fen_to_tensor(node.board.fen())[np.newaxis, :]
    policy, _ = node.model.predict(tensor)
    policy = policy.flatten()

    move_map = initialize_move_index()
    legal_moves = list(node.board.legal_moves)

    if debug:
        print("Policy Indices and Scores:")
        for i, score in enumerate(policy):
            print(f"Index {i}, Score: {score:.5f}")

    for move in legal_moves:
        move_uci = move.uci()
        move_index = move_map.get(move_uci, -1)
        if move_index == -1 or move_index >= len(policy):
            if debug:
                print(f"Move {move_uci} has no valid policy index or out of bounds index: {move_index}")
            continue

        policy_score = policy[move_index]
        if debug:
            print(f"Processing move: {move_uci}, Index: {move_index}, Policy Score: {policy_score:.5f}")

        # Further logic to create and add the child node, based on your original code
        board.push(move)
        child_board_fen = board.fen()
        board.pop()
        child = Node(state=board.fen(), model=node.model, parent=node, move=move, board=board.copy())
        node.add_child(child)
        print(f"Child added with move {move} and board state: \n{child.board}\n")
    if not node.children and debug:
        print(f"No children added for node with state:\n{node.board.fen()}")

 
def simulate(node, model, max_steps):
    board = node.board.copy()
    data = []
    print("Starting simulation with board state:", board.fen())
    steps = 0

    while not board.is_game_over() and steps < max_steps:
        policy, _ = model.predict(fen_to_tensor(board.fen())[np.newaxis, :])
        best_move = predict_move(board, policy)
        #print(f"Simulating move {best_move} at step {steps}")
        #time.sleep(.1)
        board.push(best_move)
        #print(f"Simulated move: {best_move}, Resulting board:\n {board}")
        #time.sleep(.1)#
        data.append((board.fen(), best_move.uci(), policy))
        #steps += 1
        #print("Updated board state:", board.fen())
        #time.sleep(.1)
        node.move = best_move
    
    node.board = board 
    result = combinded_eval(board)
    #print(f"Simulation Result: {result}")
    #time.sleep(.1)
    return result, data

def backpropagate(node, win_score):
    current_node = node
    path = []
    
    while current_node is not None:
        current_node.win_score += win_score
        current_node.visit_count += 1
        path.append(f"Backpropagating: Node at {current_node.state} now has win score {current_node.win_score} and visit count {current_node.visit_count}")
        #time.sleep(.1)
        current_node = current_node.parent
    
    #print("backpropagation path:")
    #for state in reversed(path):
        #print(state)

def mcts(root, depth):
    move_map = initialize_move_index()
    training_data = []
    if root.board == None:
        raise Exception("Root board is None")
    if not root.board.legal_moves:
        print("no legal moves available")
        return root, training_data

    for _ in range(depth):
        node = root
        while not node.is_leaf():
            node = select_child(node)
        if not node.board.is_game_over() and node.children == []:
            expand(node)
        win_score, game_data = simulate(node, node.model, 7)
        backpropagate(node, win_score)
        game_data = process_game_data(game_data, win_score, move_map)
        training_data.extend(game_data)

    if root.children:
        best_child = max(root.children, key=lambda x: x.win_score / x.visit_count if x.visit_count > 0 else float('-inf'))
        root = best_child
        print(f"Updating root with best child with move {root.move}")
        return best_child, training_data
    else:
        print("no Children in root after MCTS")
        return root, []

