#!/usr/bin/env python3

#Housekeeping imports
import os
import sys
import time
from datetime import datetime
#Module specific imports
import twitter as tw
import TKEYS as KEYS
#Data science imports
import pandas as pd
import numpy as np
import scipy as scy
import sklearn
#import pillow
import h5py
import tensorflow as tf
import keras
#NLP Imports
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

PYTHONIOENCODING="UTF-8"

def login():
        api = tw.Api(consumer_key = KEYS.CONSUMER_KEY, consumer_secret = KEYS.CONSUMER_SECRET, access_token_key = KEYS.ACCESS_TOKEN_KEY, access_token_secret = KEYS.ACCESS_TOKEN_SECRET, tweet_mode='extended')
        print("Connected to Twitter")
        collection(api)

def lexical_diversity(text):
        return len(set(text)) / len(text)

def collection(api):
        results = api.GetUserTimeline(include_rts=False, count=200, exclude_replies=True)
        print("Retrieving Tweets...")
        print("\n")
        mainDF = pd.DataFrame(columns=['Times', 'Tweets'])
        directory = os.getcwd()
        if os.path.isdir('./corpus') == False:
                os.makedirs('corpus')
        os.chdir(directory + '/corpus')
        new_dir = os.getcwd()
        print(f'Current Directory:\t {new_dir}\n\n')
        for tweet in results:
                fText = tweet.full_text
                fSplit = str(fText.split(' , '))
                tTime = tweet.created_at #Getting the UTC time
                mTime = time.mktime(time.strptime(tTime, "%a %b %d %H:%M:%S %z %Y"))
                eTime = int(mTime)
                #mainDF = mainDF.append({'Tweets': fSplit, 'Times': eTime}, ignore_index=True)
        ### Testing Area ###
                ld = lexical_diversity(str(tweet))
                mainDF = mainDF.append({'Tweets': fSplit, 'Times': eTime, 'LD': ld}, ignore_index=True)
                for index, r in mainDF.iterrows():
                        tweets=r['Tweets']
                        times=r['Times']
                        fname=str(user)+'_'+str(times)+'.txt'
                        corpusfile=open(new_dir+'/'+fname, 'a')
                        corpusfile.write(str(tweets))
                        #print(f"The lexical diversity of this tweet: {times} is {ld}")
                        tokenized_tweets = sent_tokenize(str(tweets))
                        #print(tokenized_tweets)
                        corpusfile.close()
                        f1name=str(user)+'.txt'
                        mainfile=open(new_dir+'/'+f1name, 'a')
                mainfile.write(str(tweets))
        print(mainDF)
        ld2 = lexical_diversity(str(mainfile))
        print(f'\nThe Lexical Diversity of all Tweets is:\t\t\t{ld2}')
        ld3 = np.mean(mainDF['LD'].describe())
        print(f'The Statistical Lexical Diversity of all Tweets is:\t{ld3}')
        ld4 = np.std(mainDF['LD'].describe())
        print(f'The StdDev of Lexical Diversity of all Tweets is:\t{ld4}')
        timeStdDev = np.std(mainDF['Times'].describe())
        print("\n\nTweets occur at this interval:\t\n")
        postInterval = int(timeStdDev)
        print(f"\t{postInterval} seconds apart.\n\n")
        if os.path.isdir('./corpus') == False:
                os.makedirs('corpus')
                #preprocessing(mainDF)
        #else:
                #preprocessing(mainDF)

#def preprocessing(mainDF):
        #example = mainDF[mainDF['Times']]['Tweets'].values[0]
        #if len(example) > 0:
        #       print(example[0])
        #       print('\nTweet:\t', example[1])


user = sys.argv[1]  
#stop_words=list(set(stopwords.words('english')))
login()
