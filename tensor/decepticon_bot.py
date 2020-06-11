#!/usr/bin/env python3


"""
This is the DECEPTICON bot. This tool is intended to automate deception
posts to Twitter for enhanced OPSEC.
"""

from decepticon_imports import *
#
#
# Thanks to Adrian, Goldar, Dreadjak, Bill, thatguy, nosirrahSec, & jferg for the help.
#
#


PYTHONIOENCODING = "UTF-8"


def login():
    """
    This module handles Authentication to Twitter.
    """
    api = tw.Api(consumer_key=keys.consumer_key,
                 consumer_secret=keys.consumer_secret,
                 access_token_key=keys.access_token_key,
                 access_token_secret=keys.access_token_secret,
                 tweet_mode='extended')
    print("\n\nConnected to Twitter\n\n")
    print("Retrieving Tweets...\n\n")
    os.chdir(DIRECTORY)
    if not path.exists(DIRECTORY+'/files.csv'):
        user_df = pd.DataFrame(columns=['Tweets', 'Times', 'LD'])
        #hopper(api, user_df)
        tokenization(api, user_df)
    else:
        user_df = pd.read_csv(os.path.join(DIRECTORY, 'files.csv'))
        #hopper(api, user_df)
        tokenization(api, user_df)


def lexical_diversity(text):
    """
    Lexical Diversity is the measurement of the variety of words used.
    """
    return len(set(str(text))) / len(text)


def tokenization(api, user_df):
    """
    Tokenization is where the tweets get broken down by word.
    The Bag of Hashtags and Mentions are also generated here as well.
    """
    results = api.GetUserTimeline(include_rts=False, count=200, exclude_replies=True)
    global BOH
    BOH = []
    global BOM
    BOM = []
    global BOL
    BOL = []
    for tweet in results:
        full_text = tweet.full_text
        full_split = str(full_text.split(' , '))
        hash_regex = re.compile(r"\s(?P<hashtag>\#\S+)\s")
        mention_regex = re.compile(r"\s(?P<mention>\@\S+)\s")
        links_regex = re.compile(r"\s(?P<links>https\:\/\/t\.co\/\S+)\s")
        links_bagger = links_regex.findall(full_split)
        mention_bagger = mention_regex.findall(full_split)
        hash_bagger = hash_regex.findall(full_split)
        if hash_bagger:
            BOH.append(hash_bagger)
        if mention_bagger:
            BOM.append(mention_bagger)
        if links_bagger:
            BOL.append(links_bagger)
    #print(f'BOM: {BOM}')
    #print(f'BOH: {BOH}')
    #print(f'BOL: {BOL}')
    hopper(api, user_df)


def hopper(api, user_df):
    """
    This module writes to the Dataframe and begins the sorting process.
    """
    results = api.GetUserTimeline(include_rts=False, count=200, exclude_replies=True)
    for tweet in results:
        full_text = tweet.full_text
        tweet_time = tweet.created_at #Getting the UTC time
        make_time = time.mktime(time.strptime(tweet_time, "%a %b %d %H:%M:%S %z %Y"))
        epoch_time = int(make_time)
        lex_div = lexical_diversity(str(tweet))
        user_df = user_df.append({'Tweets': full_text,
                                  'Times': epoch_time, 'LD': lex_div,},
                                 ignore_index=True, sort=True)
    sorter(full_text, user_df)
    user_df = user_df.drop_duplicates(subset=['Times'])


def sorter(full_text, user_df):
    """
    Creates the bag of words and triggers frequency analysis.
    """
    global INTER
    INTER = []
    global BOW
    BOW = []
    nlp = spacy.load('en_core_web_md')
    print('Beginning to tokenize tweets..\n\n')
    i = 1
    for index, r in user_df.iterrows():
        w = r['Tweets']
        INTER.append(w)
    t = Tokenizer()
    t.fit_on_texts(INTER)
    #print(t.word_counts)
    tokenized_tweets = nlp(str(INTER))
    for token in tokenized_tweets:
        if token not in STOP_WORDS:
            BOW.append(token)
    user_df.to_csv('files.csv')
    #pickle.dump(inter, open(DIRECTORY+'/full_text', 'wb'))
    for i in range(len(INTER)):
        INTER[i] = re.sub(r'\@\S+', '', full_text).split(' , ')
        INTER[i] = re.sub(r'https\:\/\/t\.co\/\S+', '', full_text).split(' , ')
    stats(user_df)
    freq_analysis()
    generate(BOW)


def stats(user_df):
    """
    This prints stats about posting and Lexical diversity
    """
    print(f"\n\nStats for tweets:\n\n")
    lex_div2 = lexical_diversity(user_df['Tweets'])
    print(f"\nThe Lexical Diversity is:\t\t\t{lex_div2}")
    lex_div3 = np.mean(user_df['LD'])
    print(f"\nThe Statistical Mean Lexical Diversity is:\t{lex_div3}")
    lex_div4 = np.std(user_df['LD'])
    print(f"\nThe StdDev of Lexical Diversity is:\t\t{lex_div4}")
    time_stddev = np.std(user_df['Times'])
    print(f"\n\nTweets occur at this interval:\t\n")
    global POST_INTERVAL
    POST_INTERVAL = int(time_stddev)
    print(f"\n\t{POST_INTERVAL} seconds apart.\n\n")
    print('\n')


def freq_analysis():
    """
    This modules conducts frequency analysis.
    """
    t = Tokenizer()
    t.fit_on_texts(BOH)
    print(f'Frequency analysis of Hashtags: {t.word_counts}')
    t = Tokenizer()
    t.fit_on_texts(BOL)
    print(f'Frequency analysis of Links: {t.word_counts}')
    t = Tokenizer()
    t.fit_on_texts(BOM)
    print(f'Frequency analysis of Mentions: {t.word_counts}')
    

def generate(BOW):
    """
    This kicks off the process of generating text.
    """
    #print(f'BOW: {BOW}')
    print('Beginning In-Depth Analysis...\n\n')
    global CHARS
    CHARS = BOW
    char_to_num = dict((c, i) for i, c in enumerate(CHARS))
    input_len = len(BOW)
    global VOCAB_LEN
    VOCAB_LEN = len(CHARS)
    print(f'Total number of characters: {input_len}')
    print(f'Total vocab: {VOCAB_LEN}')
    seq_length = 500
    x_data = []
    y_data = []
    for i in range(0, input_len - seq_length, 1):
        in_seq = BOW[i:i + seq_length]
        out_seq = BOW[i + seq_length]
        x_data.append([char_to_num[char] for char in in_seq])
        y_data.append(char_to_num[out_seq])
    n_patterns = len(x_data)
    print(f'Total Patterns: {n_patterns}')
    X = np.reshape(x_data, (n_patterns, seq_length, 1))
    X = X/float(VOCAB_LEN)
    y = np_utils.to_categorical(y_data)
    premodeling(X, y)
    trainer(x_data, CHARS)


def premodeling(X, y):
    """
    This creates the Statistical model for generating Tweets.
    """
    print('Creating the statistical model...\n\n')
    global MODEL
    MODEL = Sequential()
    filename = 'model_weights_saved.hdf5'
    global FILENAME
    FILENAME = filename
    if path.exists(filename):
        MODEL.load_weights(filename)
        modeling(X, y)
    else:
        modeling(X, y)
    
def modeling(X, y):
    MODEL.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
    MODEL.add(Dropout(0.2))
    MODEL.add(LSTM(256, return_sequences=True))
    MODEL.add(Dropout(0.2))
    MODEL.add(LSTM(256))
    MODEL.add(Dropout(0.2))
    MODEL.add(Dense(y.shape[1], activation='softmax'))
    MODEL.compile(loss='categorical_crossentropy', optimizer='adam')
    filepath = 'model_weights_saved.hdf5'
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, \
    save_best_only=True, mode='min')
    desired_callbacks = [checkpoint]
    MODEL.fit(X, y, epochs=1, batch_size=48, callbacks=desired_callbacks)
    MODEL.load_weights(FILENAME)
    MODEL.compile(loss='categorical_crossentropy', optimizer='adam')



def trainer(x_data, CHARS):
    """
    This trains the model for generating Tweets.
    """
    print('Training the model...\n\n')
    global NUM_TO_CHAR
    NUM_TO_CHAR = dict((i, c) for i, c in enumerate(str(CHARS)))
    #NUM_TO_CHAR = dict((i, c) for i, c in enumerate(str(processed_inputs)))
    start = np.random.randint(0, len(x_data) -1)
    pattern = x_data[start]
    print('Random Seed:\n\t')
    print("\"", ''.join([NUM_TO_CHAR[value] for value in pattern]), "\"\n\n")
    tweet_creator(pattern)


def tweet_creator(pattern):
    """
    This is where the magic happens and the tweets are created.
    """
    chars = sorted(BOW)
    int_to_char = dict((i, c) for i, c in enumerate(str(chars)))
    print('Initializing the process to create a tweet...\n\n')
    for i in range(3):
        x = np.reshape(pattern, (1, len(pattern), 1))
        x = x / float(VOCAB_LEN)
        prediction = MODEL.predict(x, verbose=0)
        index = np.argmax(prediction)
        result = [NUM_TO_CHAR[value] for value in pattern]
        seq_in = [int_to_char[value] for value in pattern]
        print(f'The result is:\n\n\t{result}')
        pattern.append(index)
        pattern = pattern[1:len(pattern)]


def repeater(api, full_split, epoch_time, lex_div, user_df):
    """
    If a continuation is requested, this will trigger repetition.
    """
    sleeping_interval = post_interval-(random.randint(0, 480))
    print(f"\t\t\tSleeping for {sleeping_interval} seconds...\t\t\t")
    time.sleep(sleeping_interval)
    subsequent(api, full_split, epoch_time, lex_div, user_df)


def subsequent(api, full_split, epoch_time, lex_div, user_df):
    """
    This initializes further repetition.
    """
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


#USER = sys.argv[1]
#USER = USER.lower()
nltk.download('stopwords')
DIRECTORY = os.getcwd()
if not os.path.isdir('./files'):
    os.makedirs('files')
#    os.makedirs('./files/nltk_data')
    DIRECTORY = DIRECTORY+'/files'
else:
    DIRECTORY = DIRECTORY+'/files'
STOP_WORDS = spacy.lang.en.stop_words.STOP_WORDS
login()
