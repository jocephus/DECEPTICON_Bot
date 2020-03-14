#!/usr/bin/env python3

import os
import time
import twitter as tw
import pandas as pd
import numpy as np
from datetime import datetime
import TKEYS as KEYS
import nltk
from nltk import word_tokenize
from nltk.corpus.reader import CategorizedPlaintextCorpusReader
from nltk.stem import WordNetLemmatizer, PorterStemmer
import nltk.corpus
from nltk.text import Text
from nltk import FreqDist

PYTHONIOENCODING="UTF-8"

def login():
        api = tw.Api(consumer_key = KEYS.CONSUMER_KEY, consumer_secret = KEYS.CONSUMER_SECRET, access_token_key = KEYS.ACCESS_TOKEN_KEY, access_token_secret = KEYS.ACCESS_TOKEN_SECRET, tweet_mode='extended')
        print("Connected to Twitter")
        collection(api)

def collection(api):
        results = api.GetUserTimeline(include_rts=False, count=200, exclude_replies=True)
        print("Retrieving Tweets...")
        print("\n\n\n")
        mainDF = pd.DataFrame(columns=['Times', 'Tweets'])
        for tweet in results:
                fText = tweet.full_text
                fSplit = fText.split(' , ')
                tTime = tweet.created_at #Getting the UTC time
                mTime = time.mktime(time.strptime(tTime, "%a %b %d %H:%M:%S %z %Y"))
                eTime = int(mTime)
                mainDF = mainDF.append({'Tweets': fSplit, 'Times': eTime}, ignore_index=True)
        prepreprocessing(results, tweet, mainDF, eTime, fText, fSplit, api)
        
        def prepreprocessing(results, tweet, mainDF, eTime, fText, fSplit, api):
        timeStdDev = np.std(mainDF['Times'].describe())
        print("\n\nTweets occur at this interval:\t\n")
        postInterval = int(timeStdDev)
        print(f"\t{postInterval} seconds apart.\n\n")
        directory = os.getcwd()
        if os.path.isdir('./corpus') == False:
                os.makedirs('corpus')
                preprocessing(results, tweet, mainDF, eTime, fText, fSplit, directory, api)
        else:
                preprocessing(results, tweet, mainDF, eTime, fText, fSplit, directory, api)
           
def preprocessing(results, tweet, mainDF, eTime, fText, fSplit, directory, api):
        os.chdir(directory + '/corpus')
        new_dir = os.getcwd()
        print(new_dir)
        user = input("What is your Twitter handle (less the @)\t")
        for index, r in mainDF.iterrows():
                tweets=r['Tweets']
                times=r['Times']
                fname=str(user)+'_'+str(times)+'.txt'
                corpusfile=open(new_dir+'/'+fname, 'a')
                corpusfile.write(str(tweets))
                corpusfile.close()
                f1name=str(user)+'.txt'
                mainfile=open(new_dir+'/'+f1name, 'a')
                mainfile.write(str(tweets))
                mainfile.close()
                nlpStage1(new_dir, f1name)

def nlpStage1(new_dir, f1name):
        my_corpus=CategorizedPlaintextCorpusReader('./', r'.*', cat_pattern=r'(.*)_.*')
        mfile=open(new_dir+'/'+f1name, 'r')
        fdist = FreqDist(mfile)
        w0rds = fdist.keys()
        words = w0rds[:50]
        print(words)
    
login() 
