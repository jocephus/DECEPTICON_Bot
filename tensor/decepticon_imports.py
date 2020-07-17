#!/usr/bin/env python3
  

#Housekeeping imports
import os
from os import path
import sys
import re
import time
import random
#Data science imports
import pandas as pd
import numpy as np
#AIML imports
import tensorflow as tf
from keras import backend as K
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.layers import Dense, Dropout, LSTM, Input
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import text_to_word_sequence
#NLP Imports
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
#Module specific imports
import twitter as tw
import t_keys as keys
