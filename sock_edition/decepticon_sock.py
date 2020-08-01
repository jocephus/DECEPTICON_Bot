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
    print("\n\nConnected to Twitter\n\n")
    print("Retrieving Tweets...\n\n")
    global USER_DF
    os.chdir(DIRECTORY)
    global USERS
    USERS = [] #Insert users enclosed in ' here. Example: 'Dc865_owL'
    if not path.exists(DIRECTORY+'/files_sock.csv'):
        USER_DF = pd.DataFrame(columns=['Tweets', 'Times', 'LD', 'User'])
        tokenization(api, USER_DF)
    else:
        USER_DF = pd.read_csv(os.path.join(DIRECTORY, 'files_sock.csv'))
        tokenization(api, USER_DF)


def lexical_diversity(text):
    """
    Lexical Diversity is the measurement of the variety of words used.
    """
    return len(set(str(text))) / len(text)


def tokenization(api, USER_DF):
    """
    Tokenization is where the tweets get broken down by word.
    The Bag of Hashtags and Mentions are also generated here as well.
    """
    global BOH
    BOH = []
    global BOM
    BOM = []
    global BOL
    BOL = []
    for u in USERS:
        rezzie = api.GetUserTimeline(screen_name=u, include_rts=False, count=200, exclude_replies=True)
        for tweet in rezzie:
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
    hopper(api, USER_DF)


def hopper(api, USER_DF):
    """
    This module writes to the Dataframe and begins the sorting process.
    """
    for u in USERS:
        rezzie = api.GetUserTimeline(screen_name=u, include_rts=False, count=200, exclude_replies=True)
        for tweet in rezzie:
            full_text = tweet.full_text
            tweet_time = tweet.created_at #Getting the UTC time
            make_time = time.mktime(time.strptime(tweet_time, "%a %b %d %H:%M:%S %z %Y"))
            epoch_time = int(make_time)
            lex_div = lexical_diversity(str(tweet))
            full_text = re.sub(r'\#\S+', '', str(full_text))
            full_text = re.sub(r'\@\S+', '', str(full_text))
            USER_DF = USER_DF.append({'Tweets': full_text, 'Times': epoch_time, 'LD': lex_div, 'User': u}, ignore_index=True, sort=True)
    USER_DF = USER_DF.drop_duplicates(subset=['Tweets'])
    sorter(full_text, USER_DF)


def sorter(full_text, USER_DF):
    """
    Creates the bag of words and triggers frequency analysis.
    """
    global INTER
    INTER = []
    global BOW
    BOW = []
    print('Beginning to tokenize tweets..\n\n')
    for index, r in USER_DF.iterrows():
        w = r['Tweets']
        INTER.append(w)
    t = Tokenizer()
    t.fit_on_texts(str(INTER))
    tokenized_tweets = word_tokenize(str(INTER))
    for token in tokenized_tweets:
        BOW.append(token)
    USER_DF.to_csv('files_sock.csv')
    stats(USER_DF)
    freq_analysis()
    generate(BOW)


def stats(USER_DF):
    """
    This prints stats about posting and Lexical diversity
    """
    print(f"\n\nStats for tweets:\n\n")
    lex_div2 = lexical_diversity(USER_DF['Tweets'])
    print(f"\nThe Lexical Diversity is:\t\t\t{lex_div2}")
    lex_div3 = np.mean(USER_DF['LD'])
    print(f"\nThe Statistical Mean Lexical Diversity is:\t{lex_div3}")
    lex_div4 = np.std(USER_DF['LD'])
    print(f"\nThe StdDev of Lexical Diversity is:\t\t{lex_div4}")
    time_stddev = np.std(USER_DF['Times'])
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
    X = np.reshape(x_data, (n_patterns, seq_length, 1))
    X = X/float(VOCAB_LEN)
    y = np_utils.to_categorical(y_data)
    premodeling(X, y)
    trainer(x_data, CHARS)


def regenerate(BOW):
    """
    This kicks off the process of generating text.
    """

    print('Beginning In-Depth Analysis...\n')
    global CHARS
    CHARS = BOW
    char_to_num = dict((c, i) for i, c in enumerate(CHARS))
    input_len = len(BOW)
    global VOCAB_LEN
    VOCAB_LEN = len(CHARS)
    print(f'Total number of characters: {input_len}')
    print(f'Total vocab: {VOCAB_LEN}\n')
    seq_length = 300
    x_data = []
    y_data = []
    for i in range(0, input_len - seq_length, 1):
        in_seq = BOW[i:i + seq_length]
        out_seq = BOW[i + seq_length]
        x_data.append([char_to_num[char] for char in in_seq])
        y_data.append(char_to_num[out_seq])
    n_patterns = len(x_data)
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
    filename = 'model.h5'
    global FILENAME
    FILENAME = filename
        if path.exists('model.json'):
        json_file = open('model.json', 'r')
        loaded_model_json = json_file.read()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights('model.h5')
        loaded_model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
        filepath = 'model.h5'
        checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, \
        save_best_only=True, mode='min')
        json_file.close()
        with tf.device('/gpu:0'):
            MODEL.fit(X, y, epochs=7, batch_size=8, callbacks=checkpoint)
            MODEL.load_weights(FILENAME)
            MODEL.compile(loss='categorical_crossentropy', optimizer='rmsprop')
        model_json = MODEL.to_json()
        with open('model.json', 'w') as json_file:
            json_file.write(model_json)
    else:
        modeling(X, y)

def modeling(X, y):
    MODEL.add(LSTM(160, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
    MODEL.add(Dropout(0.6))
    MODEL.add(LSTM(160, return_sequences=True))
    MODEL.add(Dropout(0.6))
    MODEL.add(LSTM(160))
    MODEL.add(Dropout(0.4))
    MODEL.add(Dense(y.shape[1], activation='softmax'))
    optimizer = RMSprop(lr=0.001)
    MODEL.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    filepath = 'model.h5'
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, \
    save_best_only=True, mode='min')
    with tf.device('/gpu:0'):
        MODEL.fit(X, y, epochs=7, batch_size=8, callbacks=checkpoint)
        MODEL.load_weights(FILENAME)
        MODEL.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    model_json = MODEL.to_json()
    with open('model.json', 'w') as json_file:
        json_file.write(model_json)


def trainer(x_data, CHARS):
    """
    This trains the model for generating Tweets.
    """
    print('Training the model...\n\n')
    global NUM_TO_CHAR
    NUM_TO_CHAR = dict((i, c) for i, c in enumerate(CHARS))
    start = np.random.randint(0, len(x_data) -1)
    pattern = x_data[start]
    print('Random Seed: Created\n\t')
    tweet_creator(pattern)


def tweet_creator(pattern):
    """
    This is where the magic happens and the tweets are created.
    """
    chars = sorted(BOW)
    int_to_char = dict((i, c) for i, c in enumerate(str(chars)))
    print('Initializing the process to create a tweet...\n\n')
    for i in range(2):
        x = np.reshape(pattern, (1, len(pattern), 1))
        x = x / float(VOCAB_LEN)
        prediction = MODEL.predict(x, verbose=1)
        index = np.argmax(prediction)
        result = [NUM_TO_CHAR[value] for value in pattern]
        seq_in = [int_to_char[value] for value in pattern]
        result = ''.join([str(itemInResult) for itemInResult in result])
        result = result.split(' , ')
        pattern.append(index)
        pattern = pattern[1:len(pattern)]
    cleanup(pattern)


def cleanup(pattern):
    """
    This module cleans up the proposed tweet
    """
    logic_one = pattern[:40]
    logic_two = str([NUM_TO_CHAR[value] for value in logic_one])
    logic_two = re.sub(r"\'\,\s\'", " ", str(logic_two))
    logic_two = re.sub(r"https\:\/\/t\.co\/\S+", "", str(logic_two))
    logic_two = re.sub(r"https\s\:\s", "", str(logic_two))
    logic_two = re.sub(r"\/\/t\.co\/\S+", "", str(logic_two))
    logic_two = re.sub(r"\[", "", str(logic_two))
    logic_two = re.sub(r"\]", "", str(logic_two))
    logic_two = re.sub(r"\@\S+", "", str(logic_two))
    logic_two = re.sub(r"\#\S+", "", str(logic_two))
    logic_two = re.sub(r"\\n", "", str(logic_two))
    logic_two = re.sub(r"\"", "", str(logic_two))
    logic_two = re.sub(r"\\", "", str(logic_two))
    logic_two = re.sub(r"\s\!", "!", str(logic_two))
    logic_two = re.sub(r"\(", "", str(logic_two))
    logic_two = re.sub(r"\)", "", str(logic_two))
    logic_two = re.sub(r"\s\:", ":", str(logic_two))
    logic_two = re.sub(r"https", "", str(logic_two))
    logic_two = re.sub(r"\'", "", str(logic_two))
    logic_two = re.sub(r"\"", "", str(logic_two))
    logic_two = re.sub(r"\,", "", str(logic_two))
    logic_two = re.sub(r"\`", "", str(logic_two))
    api.PostUpdate(str(''.join([str(itemInLogic_two) for itemInLogic_two in logic_two]) + ' #DECEPTICON @hopeconf'))
    print(str(''.join([str(itemInLogic_two) for itemInLogic_two in logic_two]) + ' #DECEPTICON @hopeconf'))
    repeater()
    
    
def repeater():
    """
    If a continuation is requested, this will trigger repetition.
    """
    sleeping_interval = POST_INTERVAL-(random.randint(0, 480))
    print(f"\t\t\tSleeping for {sleeping_interval} seconds...\t\t\t")
    sleeping_hours = (sleeping_interval / 3600)
    print(f"\t\t\tSleeping for {sleeping_hours} hours...\t\t\t")
    time.sleep(sleeping_interval)
    subsequent()


def subsequent():
    """
    This initializes further repetition.
    """
    global BOH
    BOH = []
    global BOM
    BOM = []
    global BOL
    BOL = []
    for tweet in RESULTS:
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
    tokenization(api, USER_DF)


nltk.download('stopwords')
DIRECTORY = os.getcwd()
if not os.path.isdir('./files'):
    os.makedirs('files')
    DIRECTORY = DIRECTORY+'/files'
else:
    DIRECTORY = DIRECTORY+'/files'
STOP_WORDS = set(stopwords.words('english'))
PHYSICAL_DEVICES = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(PHYSICAL_DEVICES[0], True)
except:
    pass
api = tw.Api(consumer_key=keys.consumer_key,
             consumer_secret=keys.consumer_secret,
             access_token_key=keys.access_token_key,
             access_token_secret=keys.access_token_secret,
             tweet_mode='extended', sleep_on_rate_limit=True)
MODEL = Sequential()
RESULTS = api.GetUserTimeline(include_rts=False, count=200, exclude_replies=True)
login()
