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


#### Visualizing Model Using Tensor Board 
Cloning will provide a log directory per version in which you can either run to create a "best_model" to check out the architecture through https://netron.app/ or using tensor board you can use ``tensorboard --logdir chess_v2_1/logs/(datetime)`` to bring up a local instance of tensorboard. 

#### Issues/Notes: 
As of now a lot of output and sleep calls have been commented out for strictly testing some bugs

---
TODO: 
1. Make UI 
2. ~~Redo iterability~~
3. ~~Play against stockfish (done)~~
4. ~~Change self play functionality (done)~~
5. ~~Logging (Done)~~ -> Needs Improvement 
6. ~~Store more data for visualization (Done)~~
7. Containerize 
8. Allow for building from source (Will be done with Rust, Zig, or Go rewrite)
* Mostly Rust build for Cargo and build capabilities
9. Allow for version control, versions 1 and 2 directories will stay for the time being or I will get lazy and never do it 



---
Further Goals:
* 1 Good result vs stockfish 
* Improve Eval 
* Improve Learning 
* Rewrite in rust 
    * Just needs MCTS and self play environment 
    * Adapt NNUE from https://github.com/lovechants/projects/tree/main/neural_nets/nn_rust
* Move on from chess programming soon..


