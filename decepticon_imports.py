#!/usr/bin/env python3


#Housekeeping imports
import os
from os import path
import sys
import re
import time
import random
import heapq
import pickle
#Data science imports
import pandas as pd
import numpy as np
#NLP Imports
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords, PlaintextCorpusReader
from nltk.parse.generate import generate
from nltk import ConditionalFreqDist
#Module specific imports
import twitter as tw
import t_keys as keys
