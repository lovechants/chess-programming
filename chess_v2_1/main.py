from lib import * 

tf.config.run_functions_eagerly(True) # For model training 
def main():
    model = create_model(8,256,64) # Board Size, 256, # layers
    move_map = initialize_move_index()
    opening_pos = chess.STARTING_FEN
    node = Node(opening_pos, model)
    for _ in range(30): # Number of games 
        model,node = self_simulate(node, model, 70, move_map) # 70 = number of moves allowed per game given no 50 move draw or 3 move repetition
        node.board = chess.Board(opening_pos) # Reset board 


if __name__ == "__main__":
    main()
