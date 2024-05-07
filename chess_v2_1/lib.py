import time
import math
import random 
import chess 
import numpy as np 
import tensorflow as tf 
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, BatchNormalization, ReLU, Softmax
from tensorflow.keras.models import Model
import datetime
from env import * 
from mcts import * 
from nnue import * 
import chess.engine
from keras.callbacks import ModelCheckpoint
from metric_logging import * 
import csv 
