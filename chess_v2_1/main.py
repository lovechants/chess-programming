from lib import * 
import os
import csv 
tf.config.run_functions_eagerly(True) # For model training

def main():
    model_path = 'best_model.h5'
    baseline = tf.keras.models.load_model(model_path) if os.path.exists(model_path) else None
    try:
        model = tf.keras.models.load_model(model_path)         
        print("Loaded best model for further training")
    except IOError:
        model = create_model(8,256,64) 
        print("No existing best model, creating a baseline")
    # Board Size, 256, # layers
    
    move_map = initialize_move_index()
    opening_pos = chess.STARTING_FEN
    best_val_score = float('-inf')

    for _ in range(10): # Number of games 
        node = Node(opening_pos, model)
        model,node = self_simulate(node, model, 50, move_map) # 70 = number of moves allowed per game given no 50 move draw or 3 move repetition
        node.board = chess.Board(opening_pos) # Reset board 

        if baseline:
            validation_results = evaluate_model(model, baseline, num_games=10)
            model_wins = validation_results['model_wins']
            draws = validation_results['draws']
            baseline_wins = validation_results['baseline_wins']
            validation_score = model_wins + .5 * draws #TODO robust score 
            print(f"Validation results: Model wins: {model_wins}, Draws: {draws}, Baseline wins: {baseline_wins}")
            if validation_score > best_val_score:
                best_val_score = validation_score
                model.save(model_path)
                print("New best model saved")
        else:           
            model.save(model_path)
            baseline = tf.keras.models.load_model(model_path)
            print("Baseline established")



def evaluate_model(model, baseline, num_games=2):
    results = {'model_wins': 0, 'draws': 0, 'baseline_wins': 0}
    for _ in range(num_games):
        board = chess.Board()
        while not board.is_game_over():
            current_model = model if board.turn == chess.WHITE else baseline 
            tensor = fen_to_tensor(board.fen())
            policy, _ = current_model.predict(tensor[np.newaxis,:])
            move = predict_move(board, policy)
            board.push(move)
            #time.sleep(0.01)
            print(board)    
        outcome = board.result()
        if outcome == '1-0':
            results['model_wins' if board.turn == chess.WHITE else 'baseline_wins'] += 1
        elif outcome == '0-1':
            results['baseline_wins' if board.turn == chess.WHITE else 'model_wins'] += 1
        else:
            results['draws'] += 1

    return results


def log_stockfish(game_number, model_color, model_move, stockfish_move, fen, outcome):
    with open('data_logs/stockfish_log.csv','a',newline='') as file:
        writer = csv.writer(file)
        writer.writerow([game_number, model_color, model_move, stockfish_move, fen, outcome])


def play_stockfish(best_model, n_games):
    engine_path = '/opt/homebrew/Cellar/stockfish/16.1/bin/stockfish'
    model = tf.keras.models.load_model(best_model)
    stockfish_results = {'model_wins': 0, 'draws':0, 'stockfish_wins':0}
    for i in range(n_games):
        board = chess.Board()
        engine = chess.engine.SimpleEngine.popen_uci(engine_path)

        model_color = random.choice([True, False])
        while not board.is_game_over():
            if(board.turn == chess.WHITE and model_color) or (board.turn == chess.BLACK and not model_color):
                # My model move 
                tensor = fen_to_tensor(board.fen())
                policy, _ = model.predict(tensor[np.newaxis,:])
                move = predict_move(board, policy) 
                board.push(move) 
                # push my move 
                #print(board)

                log_stockfish(i+1, "White" if model_color else "Black", move.uci(), None, board.fen(), board.result())
            else:
                result = engine.play(board, chess.engine.Limit(time=0.1))
                board.push(result.move)
                log_stockfish(i+1, "Black" if model_color else "White", None, result.move.uci(), board.fen(), board.result())
            #print(board)
           # time.sleep(1.0)
        outcome = board.result()
        if outcome == '1-0':
            if model_color:
                stockfish_results['model_wins'] += 1 
            else:
                stockfish_results['stockfish_wins'] += 1
        elif outcome == '0-1':
            if model_color:
                stockfish_results['stockfish_wins'] += 1
            else:
                stockfish_results['model_wins'] += 1 
        else:
            stockfish_results['draws'] += 1 
        print(f"Game Results: {board.result()}")
        engine.quit()
        model_wins = stockfish_results['model_wins']
        draws = stockfish_results['draws']
        stockfish_wins = stockfish_results['stockfish_wins']
    print(f"Results vs Stockfish for {n_games}\nModel wins: {model_wins}, Draws: {draws}, Stockfish wins: {stockfish_wins} ")
    

if __name__ == "__main__":
    #print("Initializing Gameplay")
    #main()
    print("Self Play Completed, Playing vs Stockfish")
    play_stockfish('best_model.h5',50)


