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
                fSplit = str(fText.split(' , '))
                tTime = tweet.created_at #Getting the UTC time
                mTime = time.mktime(time.strptime(tTime, "%a %b %d %H:%M:%S %z %Y"))
                eTime = int(mTime)
                mainDF = mainDF.append({'Tweets': fSplit, 'Times': eTime}, ignore_index=True)
        ### Testing Area ###


        print(mainDF.Tweets.value_counts())
        ### Testing Area ###
        #prepreprocessing(results, tweet, mainDF, eTime, fText, fSplit, api)

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

def lexical_diversity(text):
        return len(set(text)) / len(text)

def preprocessing(results, tweet, mainDF, eTime, fText, fSplit, directory, api):
        os.chdir(directory + '/corpus')
        new_dir = os.getcwd()
        print(new_dir)
        for index, r in mainDF.iterrows():
                tweets=r['Tweets']
                times=r['Times']
                fname=str(user)+'_'+str(times)+'.txt'
                corpusfile=open(new_dir+'/'+fname, 'a')
                corpusfile.write(str(tweets))
                ld = lexical_diversity(str(corpusfile))
                print(f"The lexical diversity of this tweet: {times} is {ld}")
                tokenized_tweets = sent_tokenize(str(tweets))
                print(tokenized_tweets)
                corpusfile.close()
                f1name=str(user)+'.txt'
                mainfile=open(new_dir+'/'+f1name, 'a')
                mainfile.write(str(tweets))
        ld = lexical_diversity(str(mainfile))
        print(f"The lexical diversity of all tweets is: {ld}")
        nlpClean(new_dir, f1name, mainDF, tweets)

# New RNN code will start here

def nlpClean(new_dir, f1name, mainDF, tweets):
        files = [open(new_dir+'/'+f, 'r').read() for f in os.listdir(new_dir)]
        all_words = []
        documents = []

        for p in files:
                documents.append(p)
                cleaned = re.sub(r'https\:\/\/t\.co\/\S{12,14}', '', p)
                recleaned = re.sub(r'[^(a-zA-Z0-9\s)]', '', cleaned)
                tokenized = word_tokenize(recleaned)
                stopped = [w for w in tokenized if not w in stop_words]
                pos = nltk.pos_tag(stopped)
                for w in pos:
                        all_words.append(w[0].lower())
        nlpStage1(new_dir, f1name, mainDF, all_words, documents, tweets, tokenized)

def nlpStage1(new_dir, f1name, mainDF, all_words, documents, tweets, tokenized):
        my_corpus=CategorizedPlaintextCorpusReader('./', r'.*', cat_pattern=r'(.*)_.*')
        filtered = []
        for w in all_words:
                if w not in stop_words:
                        filtered.append(w)
                        #refiltered = re.sub(r'[^(a-zA-Z0-9\s)]', '', str(filtered))
        #print(filtered)
        nlpStage2(all_words, tweets, filtered, documents)

def nlpStage112(tweets,):
        all_words = nltk.FreqDist()
        t2 = list(all_words.keys())[:5000]
        words = set(documents)
        features = {}
        for t in t2:
                features[t] = (t in t2)

# Trying to expand on the bag of words to do analysis on grammar and mechanics before doing the generate function
def nlpStage2(all_words, tweets, filtered, documents):
        print("2")
        featuresets = [(nlpStage112(tweets)) for (tweets) in documents]

        random.shuffle(all_words)
        print("3")
        train_data = all_words[:500]
        print(f"Train: {train_data}")
        test_data = all_words[500:]
        print("4")
        classifer = NaiveBayesClassifier.train(train_data)
        accu = classify.accuracy(classfier, test_data)
        print("5")
        print(f"Accuracy is: {accu}")
        print(classifer.show_most_informative_features(10))


user = sys.argv[1]  
#stop_words=list(set(stopwords.words('english')))
login()
