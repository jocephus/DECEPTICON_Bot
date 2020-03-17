#!/usr/bin/env python3

#Housekeeping imports
import os
import os.path
from os import path
import sys
import time
from datetime import datetime
import shutil
#Module specific imports
import twitter as tw
import TKEYS as KEYS
#Data science imports
import pandas as pd
import numpy as np
import scipy as scy
import sklearn
import h5py
import tensorflow as tf
import keras
import collections
from tensorflow.keras import layers
#NLP Imports
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

PYTHONIOENCODING="UTF-8"

def login():
        api = tw.Api(consumer_key = KEYS.CONSUMER_KEY, consumer_secret = KEYS.CONSUMER_SECRET, access_token_key = KEYS.ACCESS_TOKEN_KEY, access_token_secret = KEYS.ACCESS_TOKEN_SECRET, tweet_mode='extended')
        print("Connected to Twitter")
        directory = os.getcwd()
        directory = directory+'/files'
        if os.path.isdir('./files') == False:
                first_run(api, directory)
        else:
                collection(api, directory)

def first_run(api, directory):
        if os.path.isdir('./files') == False:
                os.makedirs('files')
                os.chdir('./files')
                os.makedirs(user)
                os.chdir(directory)
                os.makedirs('main_repo')
                print("Finsihed 1st run")
                collection(api, directory)
        else:
                collection(api, directory)

def lexical_diversity(text):
        return len(set(text)) / len(text)

def hopper1(api, directory, mainDF, userDF):
        results = api.GetUserTimeline(include_rts=False, count=200, exclude_replies=True)
        for tweet in results:
                fText = tweet.full_text
                fSplit = str(fText.split(' , '))
                tTime = tweet.created_at #Getting the UTC time
                mTime = time.mktime(time.strptime(tTime, "%a %b %d %H:%M:%S %z %Y"))
                eTime = int(mTime)
                ld = lexical_diversity(str(tweet))
                mainDF = mainDF.append({'User': user, 'Tweets': fSplit, 'Times': eTime, 'LD': ld}, ignore_index=True)
                userDF = userDF.append({'User': user, 'Tweets': fSplit, 'Times': eTime, 'LD': ld}, ignore_index=True)
                main_df(directory, mainDF, fSplit, eTime, ld)
                user_df(directory, userDF, fSplit, eTime, ld)
                subsequent(api, directory, fSplit, eTime, ld, mainDF, userDF)

def hopper2(api, directory, userDF, mainDF):
        results = api.GetUserTimeline(include_rts=False, count=200, exclude_replies=True)
        for tweet in results:
                fText = tweet.full_text
                fSplit = str(fText.split(' , '))
                tTime = tweet.created_at #Getting the UTC time
                mTime = time.mktime(time.strptime(tTime, "%a %b %d %H:%M:%S %z %Y"))
                eTime = int(mTime)
                ld = lexical_diversity(str(tweet))
                mainDF = mainDF.append({'User': user, 'Tweets': fSplit, 'Times': eTime, 'LD': ld}, ignore_index=True)
                userDF = userDF.append({'User': user, 'Tweets': fSplit, 'Times': eTime, 'LD': ld}, ignore_index=True)
                main_df(directory, mainDF, fSplit, eTime, ld)
                user_df(directory, userDF, fSplit, eTime, ld)
                subsequent(api, directory, fSplit, eTime, ld, mainDF, userDF)


def collection(api, directory):
        if path.exists(directory+'/'+user+'/'+user+'.csv') == False:
                os.chdir(directory+'/'+user+'/')
                new_dir = os.getcwd()
                print("Retrieving Tweets...")
                print("\n")
                print(f'Current Directory:\t {new_dir}\n\n')
                mainDF = pd.DataFrame(columns=['User', 'Times', 'Tweets', 'LD'])
                userDF = pd.DataFrame(columns=['User', 'Times', 'Tweets', 'LD'])
                hopper1(api, directory, mainDF, userDF)
        else:
                mainDF = pd.DataFrame(columns=['User', 'Times', 'Tweets', 'LD'])
                userDF = pd.DataFrame(columns=['User', 'Times', 'Tweets', 'LD'])
                mainDF = pd.read_csv(os.path.join(directory, 'main_repo', 'main.csv'))
                userDF = pd.read_csv(os.path.join(directory, user, user+'.csv'))
                hopper2(api, directory, userDF, mainDF)


def main_df(directory, mainDF, fSplit, eTime, ld):
                os.chdir(os.path.join(directory, 'main_repo/'))
                main_dir = os.getcwd()
                for index, r in mainDF.iterrows():
                        tweets=r['Tweets']
                        times=r['Times']
                        fname=str(user)+'_'+str(times)+'.txt'
                        corpusfile=open(main_dir+'/'+fname, 'a')
                        corpusfile.write(str(tweets))
                        tokenized_tweets = sent_tokenize(str(tweets))
                        corpusfile.close()
                        f1name='main.txt'
                        mainDF = mainDF.append({'User': user, 'Tweets': fSplit, 'Times': eTime, 'LD': ld}, ignore_index=True)
                        mainfile=open(main_dir+'/'+f1name, 'a')
                        mainfile.write(str(tweets))
                        timeStdDev = np.std(mainDF['Times'].describe())
                print('Stats for all tweets:\n\n')
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
                mainDF.to_csv(os.path.join(directory, 'main_repo', 'main.csv'))

def user_df(directory, userDF, fSplit, eTime, ld):
                os.chdir(os.path.join(directory, user))
                user_dir = os.getcwd()
                for index, r in userDF.iterrows():
                        tweets=r['Tweets']
                        times=r['Times']
                        fname=str(user)+'_'+str(times)+'.txt'
                        corpusfile=open(user_dir+'/'+fname, 'a')
                        corpusfile.write(str(tweets))
                        tokenized_tweets = sent_tokenize(str(tweets))
                        corpusfile.close()
                        f1name=str(user)+'.txt'
                        userDF = userDF.append({'User': user, 'Tweets': fSplit, 'Times': eTime, 'LD': ld}, ignore_index=True)
                        mainfile=open(user_dir+'/'+f1name, 'a')
                        mainfile.write(str(tweets))
                        timeStdDev = np.std(userDF['Times'].describe())
                print(f"Stats for {user}'s tweets:\n\n")
                ld2 = lexical_diversity(str(mainfile))
                print(f'\nThe Lexical Diversity of all Tweets is:\t\t\t{ld2}')
                ld3 = np.mean(userDF['LD'].describe())
                print(f'The Statistical Lexical Diversity of all Tweets is:\t{ld3}')
                ld4 = np.std(userDF['LD'].describe())
                print(f'The StdDev of Lexical Diversity of all Tweets is:\t{ld4}')
                timeStdDev = np.std(userDF['Times'].describe())
                print("\n\nTweets occur at this interval:\t\n")
                postInterval = int(timeStdDev)
                print(f"\t{postInterval} seconds apart.\n\n")
                userDF.to_csv(os.path.join(directory, user, user+'.csv'))

def subsequent(api, directory, fSplit, eTime, ld, mainDF, userDF):
        results = api.GetUserTimeline(include_rts=False, count=200, exclude_replies=True)
        os.chdir(directory)
        shutil.move(os.path.join(directory, 'main_repo', 'main.csv'), os.path.join(directory, 'main.csv'))
        shutil.move(os.path.join(directory, user, user+'.csv'), os.path.join(directory, user+'.csv'))
        os.chdir(directory)
        userDF = pd.read_csv(user+'.csv')
        mainDF = pd.read_csv('main.csv')
        print("Retrieving Tweets...")
        print("\n")
        for tweet in results:
                tweets=mainDF['Tweets']
                times=mainDF['Times']
                fText = tweet.full_text
                fSplit = str(fText.split(' , '))
                tTime = tweet.created_at #Getting the UTC time
                mTime = time.mktime(time.strptime(tTime, "%a %b %d %H:%M:%S %z %Y"))
                eTime = int(mTime)
                ld = lexical_diversity(str(tweet))
                mainDF = mainDF.append({'User': user, 'Tweets': fSplit, 'Times': eTime, 'LD': ld}, ignore_index=True)
                mainDF = mainDF.drop_duplicates(keep='first')
                main_df(directory, mainDF, fSplit, eTime, ld)
                userDF = userDF.append({'User': user, 'Tweets': fSplit, 'Times': eTime, 'LD': ld}, ignore_index=True)
                userDF = userDF.drop_duplicates(keep='first')
                user_df(directory, userDF, fSplit, eTime, ld)
                mainfile=open(os.path.join(directory, 'main_repo', 'main.txt'), 'a')
                mainfile.write(str(tweets))
        print('Updated Stats for all tweets:\n\n')
        ld2 = lexical_diversity(str(mainfile))
        print(f'\nThe Lexical Diversity of all Tweets is:\t\t\t{ld2}')
        ld3 = np.mean(mainDF['LD'].describe())
        print(f'The Updated Statistical Lexical Diversity of all Tweets is:\t{ld3}')
        ld4 = np.std(mainDF['LD'].describe())
        print(f'The Updated StdDev of Lexical Diversity of all Tweets is:\t{ld4}')
        timeStdDev = np.std(mainDF['Times'].describe())
        print("\n\nTweets occur at this Updated interval:\t\n")
        postInterval = int(timeStdDev)
        print(f"\t{postInterval} seconds apart.\n\n")
        userDF.to_csv('user.csv')
        mainDF.to_csv('main.csv')
        shutil.move(os.path.join(directory, 'main.csv'), os.path.join(directory, 'main_repo', 'main.csv'))
        shutil.move(os.path.join(directory, user+'.csv'), os.path.join(directory, user, user+'.csv'))
        os.chdir(directory)
        gonogo(api, directory, fSplit, eTime, ld, mainDF, userDF)

def gonogo(api, directory, fSplit, eTime, ld, mainDF, userDF):
        gonogo = input("Continue? (Y/N)")
        if gonogo.lower() == 'y':
                print("Sleeping for 4 hours")
                time.sleep(14400)
                subsequent()
        else:
                print("Goodbye")
                exit()

user = sys.argv[1]
user = user.lower()  
login()
