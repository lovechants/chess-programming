import logging 
import csv

def create_training_log():
    with open ('data_logs/training_metrics.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Policy Loss', 'Value Loss'])

def create_game_log():
    with open('data_logs/game_log.csv','w',newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Game Number', 'Turn', 'Move', 'Fen'])

def create_stockfish_log():
    with open('data_logs/stockfish_log.csv','w',newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Game Number', 'Turn', 'Move', 'Fen'])


def log_stockfish(game_number, model_color, model_move, stockfish_move, fen, outcome):
    with open('data_logs/stockfish_log.csv','a',newline='') as file:
        writer = csv.writer(file)
        writer.writerow([game_number, model_color, model_move.uci(), stockfish_move.uci(), fen, outcome])


create_training_log()
create_game_log() 
create_stockfish_log()
