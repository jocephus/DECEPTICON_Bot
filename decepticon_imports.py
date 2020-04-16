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
#import h5py
#import tensorflow as tf
#import keras
#import collections
#from tensorflow.keras import layers
#from keras.models import Sequential
#from keras.layers import Dense, LSTM, Dropout, Bidirectional, Embedding
#from keras.callbacks import ModelCheckpoint
#NLP Imports
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from textblob import Word
#Bag of Words Support
#import scipy as scy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
#Module specific imports
import twitter as tw
import TKEYS as KEYS
