#!/usr/bin/env python3

#Housekeeping imports
import os
import os.path
from os import path
import sys
import time
import re
from datetime import datetime
import random
#Module specific imports
import twitter as tw
import TKEYS as KEYS
#Data science imports
import pandas as pd
import numpy as np
#import scipy as scy
import sklearn
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
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
#
#
# Thanks to Goldar, Dreadjak, Bill, thatguy, nosirrahSec, & jferg for the help.
#
#

PYTHONIOENCODING="UTF-8"

def login():
        api = tw.Api(consumer_key = KEYS.CONSUMER_KEY, consumer_secret = KEYS.CONSUMER_SECRET, access_token_key = KEYS.ACCESS_TOKEN_KEY, access_token_secret = KEYS.ACCESS_TOKEN_SECRET, tweet_mode='extended')
        print("\n\nConnected to Twitter\n\n")
        print("Retrieving Tweets...\n")
        os.chdir(directory)
        if path.exists(directory+'/files.csv') == False:
                userDF = pd.DataFrame(columns=['User', 'Tweets', 'Times', 'LD', 'Stemmed', 'Lemmerized'])
                #hopper(api, userDF)
                tokenization(api, userDF)
        else:
                userDF = pd.read_csv(os.path.join(directory, 'files.csv'))
                #hopper(api, userDF)
                tokenization(api, userDF)

def lexical_diversity(text):
        return len(set(text)) / len(text)

def word_extraction(sentence):
        ignore = ['a', 'the', 'is']
        words = re.sub("[^\w]", " ", sentence).split()
        cleaned_text = [w.lower() for w in words if w not in ignore]
        return cleaned_text

def tokenization(api, userDF):
        results = api.GetUserTimeline(include_rts=False, count=200, exclude_replies=True)
        words = []
        for tweet in results:
                fText = tweet.full_text
                fSplit = str(fText.split(' , '))
                wext = word_extraction(fSplit)
                words.extend(wext)
        words = sorted(list(set(words)))
        #print(words)
        hopper(words, api, userDF)

def hopper(words, api, userDF):
        results = api.GetUserTimeline(include_rts=False, count=200, exclude_replies=True)
        ps = PorterStemmer()
        bank = []
        lemm = []
        stem = []
        for tweet in results:
                fText = tweet.full_text
                fSplit = str(fText.split(' , '))
                tTime = tweet.created_at #Getting the UTC time
                mTime = time.mktime(time.strptime(tTime, "%a %b %d %H:%M:%S %z %Y"))
                eTime = int(mTime)
                ld = lexical_diversity(str(tweet))
                tokenized_tweets = sent_tokenize(fSplit)
                for w in words:
                        if w not in stop_words1:
                                bank.append(w)
                for w in bank:
                        rootWord=ps.stem(w)
                        stem.append(rootWord)
                for i in bank:
                        word1 = Word(i).lemmatize("n")
                        word2 = Word(word1).lemmatize("v")
                        word3 = Word(word2).lemmatize("a")
                        lemm.append(Word(word3).lemmatize())
                userDF = userDF.append({'User': user, 'Tweets': fSplit, 'Times': eTime, 'LD': ld, 'Stemmed': rootWord, 'Lemmerized': lemm}, ignore_index=True, sort=True)
                userDF = userDF.drop_duplicates(subset=['Times'])
        userDF = userDF.drop_duplicates(subset=['Times'])
        userDF.to_csv('files.csv')
        for index, r in userDF.iterrows():
                tweets=r['Tweets']
                times=r['Times']
                stem=r['Stemmed']
                lemm=r['Lemmerized']
                fname=str(user)+'_'+str(eTime)+'.txt'
                corpusfile=open(fname, 'a')
                corpusfile.write('Time: '+str(times))
                corpusfile.write('\nTweets:'+str(tweets))
                corpusfile.write('\nStemmed:'+str(stem))
                corpusfile.write('\nLemmerized:'+str(lemm))
                corpusfile.close()
        print(f"Stats for {user}'s tweets:\n\n")
        ld2 = lexical_diversity(userDF['Tweets'])
        print(f"\nThe Lexical Diversity of {user}'s Tweets is:\t\t\t{ld2}")
        ld3 = np.mean(userDF['LD'])
        print(f"The Statistical Mean Lexical Diversity of {user}'s Tweets is:\t{ld3}")
        ld4 = np.std(userDF['LD'])
        print(f"The StdDev of Lexical Diversity of {user}'s Tweets is:\t\t{ld4}")
        timeStdDev = np.std(userDF['Times'])
        print(f"\n\n{user}'s Tweets occur at this interval:\t\n")
        postInterval = int(timeStdDev)
        print(f"\t{postInterval} seconds apart.\n\n")
        bagOWords1(api, fSplit, eTime, ld, userDF)

def gonogo(api, fSplit, eTime, ld, userDF):
        gonogo = input("Continue? (Y/N)")
        if gonogo.lower() == 'y':
                print("Sleeping for 4 hours")
                time.sleep(10)
                subsequent(api, fSplit, eTime, ld, userDF)
        else:
                print("Goodbye")
                exit()

def repeater(api, fSplit, eTime, ld, userDF, postInterval):
        sleeping_interval = postInterval-(random.randint(0,480))
        print(f"\t\t\tSleeping for {sleeping_interval} seconds...\t\t\t")
        time.sleep(sleeping_interval)
        subsequent(api, fSplit, eTime, ld, userDF)

def bagOWords1(api, fSplit, eTime, ld, userDF):
        print("Beginning NLP Analysis...")
        print('\n\nOnto the fun stuff...\n\n')
        userDF = pd.read_csv('files.csv')
        for index, r in userDF.iterrows():
                lemmerized=r['Lemmerized']
                lemmerized = [str(lemmerized)]
                cVectorizer.fit(lemmerized)
        print(cVectorizer.vocabulary_)
        vector = cVectorizer.transform(lemmerized)
        print(vector.shape)
        print(type(vector))
        print(vector.toarray())
        bagOWords2(api, fSplit, eTime, ld, userDF)

def bagOWords2(api, fSplit, eTime, ld, userDF):
        print('\n\n\t\tStage 2 NLP Analysis...\n\n')
        userDF = pd.read_csv('files.csv')
        for index, r in userDF.iterrows():
                lemmerized=r['Lemmerized']
                lemmerized = [str(lemmerized)]
                tVectorizer.fit(lemmerized)
        print(tVectorizer.vocabulary_)
        print(tVectorizer.idf_)
        vector = tVectorizer.transform(lemmerized)
        print(vector.shape)
        print(type(vector))
        print(vector.toarray())
        bagOWords3(api, fSplit, eTime, ld, userDF)

def bagOWords3(api, fSplit, eTime, ld, userDF):
        print('\n\n\t\tStage 3 NLP Analysis...\n\n')
        print('t\tLemmerized NLP Analysis...\n\n')
        userDF = pd.read_csv('files.csv')
        for index, r in userDF.iterrows():
                lemmerized=r['Lemmerized']
                lemmerized = [str(lemmerized)]
                tVectorizer.fit(lemmerized)
        vector = hVectorizer.transform(lemmerized)
        print(vector.shape)
        print(type(vector))
        print(vector.toarray())
        gonogo(api, fSplit, eTime, ld, userDF)

def subsequent(api, fSplit, eTime, ld, userDF):
        results = api.GetUserTimeline(include_rts=False, count=200, exclude_replies=True)
        print("Retrieving Tweets...")
        print("\n")
        hopper(api, userDF)
        userDF = userDF1.drop_duplicates(subset=['Times'])
        print('\n\nUpdated Stats for all tweets:\n\n')
        ld2 = lexical_diversity(userDF['Tweets'])
        print(f'\nThe Lexical Diversity of Tweets is:\t\t\t\t\t{ld2}')
        ld3 = np.mean(userDF['LD'])
        print(f'The Updated Statistical Lexical Diversity of Tweets is:\t\t\t{ld3}')
        ld4 = np.std(userDF['LD'])
        print("\n\nTweets occur at this Updated interval:\t\n")
        timeStdDev = np.std(userDF['Times'])
        postInterval = int(timeStdDev)
        print(f"\t{postInterval} seconds apart.\n\n")
        userDF = userDF.drop_duplicates()
        userDF.to_csv('user.csv')
        hopper(api, userDF)


user = sys.argv[1]
user = user.lower()
directory = os.getcwd()
if os.path.isdir('./files') == False:
        os.makedirs('files')
        directory = directory+'/files'
else:
        directory = directory+'/files'
stop_words1 = set(stopwords.words('english'))
cVectorizer = CountVectorizer()
tVectorizer = TfidfVectorizer()
hVectorizer = HashingVectorizer(n_features=20)
login() 
