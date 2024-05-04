from neural_net import * 
from pgn_processing import * 
from game_eval import * 
from movement import * 
import math
import chess
import random 
from reinforcement import *
import time 
from exploratory import * 
# Node 
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
"""
def expand(node):
    board = node.board
    print(f"Expanding node for board state: {node.state}")
    for move in board.legal_moves:
        board.push(move)
        child = Node(state=board.fen(),model = create_model(8,256,2), parent=node, move=move, board=board)
        node.add_child(child)
        board.pop()
    print(f"Added {len(node.children)} children.")
"""

def apply_temperature(policy, temp=1.0):
    if temp == 0:
        new_policy = np.zeros_like(policy)
        new_policy[np.argmax(policy)] = 1 
        return new_policy
    else:
        softened_policy = np.exp(np.log(policy + 1e-6) / temp)
        softened_policy /= softened_policy.sum()
        return softened_policy

def add_noise(policy, noise=.06):
    noise = np.random.randn(*policy.shape) * noise
    noise_policy = policy + noise
    noise_policy = np.clip(noise_policy, 0,1)
    noise_policy /= noise_policy.sum()

    return noise_policy
'''
def expand(node):
    if node.board is None:
        raise Exception("Trying to expand a node without a valid board.")
    board = node.board.copy()
    print(f"Expanding node for board state: {board.fen()} with {len(list(board.legal_moves))} legal moves.")
    move_map = initialize_move_index()
    tensor = fen_to_tensor(board.fen())[np.newaxis,:]
    policy, _ = node.model.predict(tensor)
    policy = apply_temperature(policy)
    policy = add_noise(policy)
    if len(list(board.legal_moves)) == 0:
        print("No legal moves available to expand.")
        return
    for move in board.legal_moves:
        move_uci = move.uci()
        move_index = move_map.get(move_uci, -1)
        if move_index == -1 or move_index >= len(policy):
            continue
        board.push(move)
        new_board = board.copy()
        child = Node(state=board.fen(), model = node.model, parent=node, move = move, board=new_board)
        node.add_child(child)
        print(f"Child added with move {move} and board state: \n{child.board}\n")
        board.pop()
    print(f"Added {len(node.children)} children")
    if not node.children:
        print(f"No children added for node with state: {node.board.fen()}")

def simulate(node, model, max_steps=30):
    repeat_count = {}
    seen_positions = set()
    game_data = []
    board = chess.Board(node.state)
    steps = 0
    move_history = []
    while not board.is_game_over(claim_draw=True) and steps < max_steps:
        current_fen = board.fen()
        repeat_count[current_fen] = repeat_count.get(current_fen, 0)
        if current_fen in seen_positions:
            print("Repeated States")
            break
        if repeat_count[current_fen] >= 3:
            print("Threfold reptition detected")
            break
        seen_positions.add(current_fen)
        tensor = fen_to_tensor(board.fen())
        policy, _ = model.predict(tensor[np.newaxis, :])
        
        best_move = predict_move(board, policy)
        if best_move in board.legal_moves:
            move_history.append(best_move.uci())
            board.push(best_move)
            node.move = best_move
            node.board = board.copy()
            node.state = board.fen()

            game_data.append((board.fen(), best_move.uci(), policy))
        else:
            print("Illegal Move, picking random move")
            move = random.choice(list(board.legal_moves))
            move_history.append(move.uci())
            board.push(move)
            node.move = move 
            node.board = board.copy()
            node.state = board.fen()
            game_data.append((board.fen(), move.uci(), policy))
        if board.can_claim_threefold_repetition() or board.can_claim_fifty_moves():
            print("Draw by reptetition")
            break
        time.sleep(.01)
        steps += 1
    result = combinded_eval(board) 
    print(f"Simulation Result: {result}")
    return result, game_data

def expand(node, debug=False):
    if node.board is None:
        raise Exception("trying to expand a node without a balid board object")
    legal_moves = list(node.board.legal_moves)
    if not legal_moves:
        print("no legal moves available to expand")
        return
    board = node.board.copy()
    print(f"Expanding node for board state: {board.fen()} with {len(legal_moves)} legal moves")
    move_map = initialize_move_index()
    tensor = fen_to_tensor(board.fen())[np.newaxis, :]
    policy, _ = node.model.predict(tensor)
    #policy = apply_temperature(policy)
    #policy = add_noise(policy)
    
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
    for move in legal_moves:
        move_uci = move.uci()
        move_index = move_map.get(move_uci, -1)
        if move_index == -1 or move_index >= len(policy):
            print(f"Move {move_uci} has no valid policy index")
            continue

        print(f"Processing move: {move_uci}, Index: {move_index}, Policy Score: {policy[0][move_index]}")
        board.push(move)
        child_board_fen = board.fen()
        board.pop()
        child = Node(state=child_board_fen, model=node.model, parent=node, move=move, board=board.copy())
        node.add_child(child)
        print(f"Child added with move {move} and board state: \n{child.board}\n")
    print(f"Added {len(node.children)} children")
    if not node.children:
        print(f"No children added for node with state:\n{node.board}\n")
'''


def expand(node, debug=False):
    if node.board is None:
        raise Exception("trying to expand a node without a balid board object")
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
        print(f"Policy output: {policy}")
        best_move = predict_move(board, policy)
        print(f"Simulating move {best_move} at step {steps}")
        board.push(best_move)
        print(f"Simulated move: {best_move}, Resulting board:\n {board}")
        print(f"Turn after move {node.move}: {'White' if node.board.turn == chess.WHITE else 'Black'}")
        #time.sleep(.1)
        data.append((board.fen(), best_move.uci(), policy))
        steps += 1
        print("Updated board state:", board.fen())
        node.move = best_move
    #board.push(best_move)
    #node.move = best_move
    node.board = board 
    result = combinded_eval(board)
    print(f"Simulation Result: {result}")
    return result, data


def backpropagate(node, win_score):
    current_node = node
    path = []
    while current_node is not None:
        current_node.win_score += win_score
        current_node.visit_count += 1
        path.append(f"Backpropagating: Node at {current_node.state} now has win score {current_node.win_score} and visit count {current_node.visit_count}")
        current_node = current_node.parent
    print("backpropagation path:")
    for state in reversed(path):
        print(state)
"""
def mcts(root, n):
    move_map = initialize_move_index()
    training_data = []
    for i in range(n):
        node = root
        print(f"Starting iteration {i+1}")
        while node.children:
            node = select_child(node)
        print(f"Selected node for expansion with state: {node.state}")
        if node.visit_count == 0:
            expand(node)


        win_score, game_data = simulate(node, node.model, 30)

        backpropagate(node, win_score)
        print(f"Iteration {i+1}/{n}: Root node win score {root.win_score}, visits {root.visit_count}")
        game_result = win_score
        processed_data = process_game_data(game_data, game_result, move_map)
        training_data.extend(processed_data)
        if root.children:
            best_child = max(root.children, key=lambda x: x.win_score / x.visit_count if x.visit_count > 0 else float('-inf'))
            root.state = best_child.state
            root.children = best_child.children
            root.visit_count = best_child.visit_count
            root.win_score = best_child.win_score
            root.move = best_child.move
            root.board = best_child.board.copy()
            print(f"Updating root to best child state: {root.move}")
        
    return root, training_data
"""
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

