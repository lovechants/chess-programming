# chess-programming
All chess models I make good or bad

---

## Current readable model version 2.1 

Version 2.1 now plays against stockfish.

#### Current Looping Structure:
1. Initialize a model if no .h5 model file found 
2. Current model plays against itself, and then trains vs a baseline model 
    * If baseline does not exist baseline eval = $-inf$
    * If baseline does exist evaluate current model performance after N games versus the baseline
3. Continue training given input values
4. Play stockfish for N games

#### Issues/Notes: 
As of now a lot of output and sleep calls have been commented out for strictly testing some bugs

---
TODO: 
1. Make UI 
2. ~~Redo iterability~~
3. ~~Play against stockfish (done)~~
4. ~~Change self play functionality (done)~~
5. Better logging (in-progress)
6. Store more data for visualization (in-progress)
7. Rewrite in rust 
* Just needs MCTS and self play environment 
* Adapt NNUE from https://github.com/lovechants/projects/tree/main/neural_nets/nn_rust
8. Containerize 
