#!/usr/bin/env python3


from decepticon_imports import *
#
#
# Thanks to Adrian, Goldar, Dreadjak, Bill, thatguy, nosirrahSec, & jferg for the help.
#
#


PYTHONIOENCODING = "UTF-8"


def login():
    api = tw.Api(consumer_key=keys.consumer_key,
                 consumer_secret=keys.consumer_secret,
                 access_token_key=keys.access_token_key,
                 access_token_secret=keys.access_token_secret,
                 tweet_mode='extended')
    print("\n\nConnected to Twitter\n\n")
    print("Retrieving Tweets...\n")
    os.chdir(DIRECTORY)
    if not path.exists(DIRECTORY+'/files.csv'):
        user_df = pd.DataFrame(columns=['User', 'Tweets', 'Times', 'LD', 'Stemmed', 'Lemmerized'])
        #hopper(api, user_df)
        tokenization(api, user_df)
    else:
        user_df = pd.read_csv(os.path.join(DIRECTORY, 'files.csv'))
        #hopper(api, user_df)
        tokenization(api, user_df)


def lexical_diversity(text):
    return len(set(text)) / len(text)


def word_extraction(sentence):
    ignore = ['a', 'the', 'is']
    words = re.sub("[^\w]", " ", sentence).split()
    cleaned_text = [w.lower() for w in words if w not in ignore]
    return cleaned_text


def tokenization(api, user_df):
    results = api.GetUserTimeline(include_rts=False, count=200, exclude_replies=True)
    words = []
    for tweet in results:
        f_text = tweet.full_text
        f_split = str(f_text.split(' , '))
        wext = word_extraction(f_split)
        words.extend(wext)
    words = sorted(list(set(words)))
    #print(words)
    hopper(words, api, user_df)


def hopper(words, api, user_df):
    results = api.GetUserTimeline(include_rts=False, count=200, exclude_replies=True)
    port_stemmer = PorterStemmer()
    bank = []
    lemm = []
    stem = []
    for tweet in results:
        full_text = tweet.full_text
        full_split = str(full_text.split(' , '))
        tweet_time = tweet.created_at #Getting the UTC time
        make_time = time.mktime(time.strptime(tweet_time, "%a %b %d %H:%M:%S %z %Y"))
        epoch_time = int(make_time)
        lex_div = lexical_diversity(str(tweet))
        tokenized_tweets = sent_tokenize(full_split)
        for w in words:
            if w not in STOP_WORDS:
                bank.append(w)
        for w in bank:
            rootWord = port_stemmer.stem(w)
            stem.append(rootWord)
        for i in bank:
            word1 = Word(i).lemmatize("n")
            word2 = Word(word1).lemmatize("v")
            word3 = Word(word2).lemmatize("a")
            lemm.append(Word(word3).lemmatize())
        user_df = user_df.append({'User': USER, 'Tweets': full_split,
                                  'Times': epoch_time, 'LD': lex_div,
                                  'Stemmed': rootWord,
                                  'Lemmerized': lemm},
                                 ignore_index=True, sort=True)
        user_df = user_df.drop_duplicates(subset=['Times'])
    user_df = user_df.drop_duplicates(subset=['Times'])
    user_df.to_csv('files.csv')
    for index, r in user_df.iterrows():
        tweets = r['Tweets']
        times = r['Times']
        stem = r['Stemmed']
        lemm = r['Lemmerized']
        f_name = str(USER)+'_'+str(epoch_time)+'.txt'
        corpusfile = open(f_name, 'a')
        corpusfile.write('Time: '+str(times))
        corpusfile.write('\nTweets:'+str(tweets))
        corpusfile.write('\nStemmed:'+str(stem))
        corpusfile.write('\nLemmerized:'+str(lemm))
        corpusfile.close()
    print(f"Stats for {USER}'s tweets:\n\n")
    lex_div2 = lexical_diversity(user_df['Tweets'])
    print(f"\nThe Lexical Diversity of {USER}'s Tweets is:\t\t\t{lex_div2}")
    lex_div3 = np.mean(user_df['LD'])
    print(f"The Statistical Mean Lexical Diversity of {USER}'s Tweets is:\t{lex_div3}")
    lex_div4 = np.std(user_df['LD'])
    print(f"The StdDev of Lexical Diversity of {USER}'s Tweets is:\t\t{lex_div4}")
    time_stddev = np.std(user_df['Times'])
    print(f"\n\n{USER}'s Tweets occur at this interval:\t\n")
    post_interval = int(time_stddev)
    print(f"\t{post_interval} seconds apart.\n\n")
    bagOWords1(api, full_split, epoch_time, lex_div, user_df)
    
    def gonogo(api, full_split, epoch_time, lex_div, user_df):
    gonogo = input("Continue? (Y/N)")
    if gonogo.lower() == 'y':
        print("Sleeping for 4 hours")
        time.sleep(10)
        subsequent(api, full_split, epoch_time, lex_div, user_df)
    else:
        print("Goodbye")
        exit()


def repeater(api, full_split, epoch_time, lex_div, user_df, post_interval):
    sleeping_interval = post_interval-(random.randint(0, 480))
    print(f"\t\t\tSleeping for {sleeping_interval} seconds...\t\t\t")
    time.sleep(sleeping_interval)
    subsequent(api, full_split, epoch_time, lex_div, user_df)


def bagOWords1(api, full_split, epoch_time, lex_div, user_df):
    print("Beginning NLP Analysis...")
    print('\n\nOnto the fun stuff...\n\n')
    user_df = pd.read_csv('files.csv')
    print('\t\tNLP Stage 1\n')
    for index, r in user_df.iterrows():
        lemmerized = r['Lemmerized']
        lemmerized = [str(lemmerized)]
        C_VECTORIZER.fit(lemmerized)
    print(C_VECTORIZER.vocabulary_)
    vector = C_VECTORIZER.transform(lemmerized)
    print(vector.shape)
    print(type(vector))
    victory_one = vector.toarray()
    victory_one = np.array(victory_one)
    print(victory_one)
    print('\t\tNLP Stage 2\n')
    for index, r in user_df.iterrows():
        lemmerized = r['Lemmerized']
        lemmerized = [str(lemmerized)]
        T_VECTORIZER.fit(lemmerized)
        print(T_VECTORIZER.vocabulary_)
    print(T_VECTORIZER.idf_)
    vector = T_VECTORIZER.transform(lemmerized)
    print(vector.shape)
    print(type(vector))
    victory_two = vector.toarray()
    victory_two = np.array(victory_two)
    print(victory_two)
    print('\t\tNLP Stage 3\n')
    for index, r in user_df.iterrows():
        lemmerized = r['Lemmerized']
        lemmerized = [str(lemmerized)]
        T_VECTORIZER.fit(lemmerized)
    vector = H_VECTORIZER.transform(lemmerized)
    print(vector.shape)
    print(type(vector))
    victory_three = vector.toarray()
    victory_three = np.array(victory_three)
    print(victory_three)
    generation(api, full_split, epoch_time, lex_div, user_df, victory_one, victory_two, victory_three)


def generation(api, full_split, epoch_time, lex_div, user_df, victory_one, victory_two, victory_three):
    for index, r in user_df.iterrows():
        bag_vector = np.zeros(len(victory_one))
        for v in victory_one:
            for i, word in enumerate(v):
                if word == v.any():
                    bag_vector[i] += 1
                    print("{0}\n{1}\n".format(victory_one, np.array(bag_vector)))


def subsequent(api, full_split, epoch_time, lex_div, user_df):
    results = api.GetUserTimeline(include_rts=False, count=200, exclude_replies=True)
    print("Retrieving Tweets...")
    print("\n")
    hopper(api, user_df)
    user_df = user_df1.drop_duplicates(subset=['Times'])
    print('\n\nUpdated Stats for all tweets:\n\n')
        lex_div2 = lexical_diversity(user_df['Tweets'])
    print(f'\nThe Lexical Diversity of Tweets is:\t\t\t\t\t{lex_div2}')
    lex_div3 = np.mean(user_df['LD'])
    print(f'The Updated Statistical Lexical Diversity of Tweets is:\t\t\t{lex_div3}')
    lex_div4 = np.std(user_df['LD'])
    print("\n\nTweets occur at this Updated interval:\t\n")
    time_stddev = np.std(user_df['Times'])
    post_interval = int(time_stddev)
    print(f"\t{post_interval} seconds apart.\n\n")
    user_df = user_df.drop_duplicates()
    user_df.to_csv('user.csv')
    hopper(api, user_df)


USER = sys.argv[1]
USER = USER.lower()
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
DIRECTORY = os.getcwd()
if not os.path.isdir('./files'):
    os.makedirs('files')
    DIRECTORY = DIRECTORY+'/files'
else:
    DIRECTORY = DIRECTORY+'/files'
STOP_WORDS = set(stopwords.words('english'))
C_VECTORIZER = CountVectorizer()
T_VECTORIZER = TfidfVectorizer()
H_VECTORIZER = HashingVectorizer(n_features=20)
login()
