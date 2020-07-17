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
    if not path.exists(DIRECTORY+'/files.csv'):
        USER_DF = pd.DataFrame(columns=['Tweets', 'Times', 'LD'])
        tokenization(api, USER_DF)
    else:
        USER_DF = pd.read_csv(os.path.join(DIRECTORY, 'files.csv'))
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
    hopper(api, USER_DF)


def hopper(api, USER_DF):
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
        USER_DF = USER_DF.append({'Tweets': full_text, 'Times': epoch_time, 'LD': lex_div,}, ignore_index=True, sort=True)
    sorter(full_text, USER_DF)
    USER_DF = USER_DF.drop_duplicates(subset=['Times'])


def sorter(full_text, USER_DF):
    """
    Creates the bag of words and triggers frequency analysis.
    """
    global INTER
    INTER = []
    global BOW
    BOW = []
    print('Beginning to tokenize tweets..\n\n')
    i = 1
    for index, r in USER_DF.iterrows():
        w = r['Tweets']
        INTER.append(w)
    t = Tokenizer()
    t.fit_on_texts(INTER)
    tokenized_tweets = word_tokenize(str(INTER))
    for token in tokenized_tweets:
        #if token not in STOP_WORDS:
            #BOW.append(token)
        BOW.append(token)
    USER_DF.to_csv('files.csv')
    for i in range(len(INTER)):
        INTER[i] = re.sub(r'\@\W+', '', str(INTER[i])).split(' , ')
        INTER[i] = re.sub(r'\s\S+t\.co\/\S+\s', '', str(INTER[i])).split(' , ')
        INTER[i] = re.sub(r'\/+', '', str(INTER[i])).split(' , ')
        INTER[i] = re.sub(r'\\+', '', str(INTER[i])).split(' , ')
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
    #print(f'Total Patterns: {n_patterns}')
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
    MODEL.add(Dropout(0.6))
    MODEL.add(LSTM(256, return_sequences=True))
    MODEL.add(Dropout(0.6))
    MODEL.add(LSTM(256))
    MODEL.add(Dropout(0.6))
    MODEL.add(Dense(y.shape[1], activation='softmax'))
    optimizer = RMSprop(lr=0.001)
    MODEL.compile(loss='categorical_crossentropy', optimizer='adam')
    filepath = 'model_weights_saved.hdf5'
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, \
    save_best_only=True, mode='min')
    with tf.device('/gpu:0'):
        MODEL.fit(X, y, epochs=3, batch_size=16, callbacks=checkpoint)
        MODEL.load_weights(FILENAME)
        MODEL.compile(loss='categorical_crossentropy', optimizer='adam')


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
    #print("\"", ''.join([NUM_TO_CHAR[value] for value in pattern]), "\"\n\n")
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
        #print(f'The result is:\n\n\t{result}\n')
        pattern.append(index)
        pattern = pattern[1:len(pattern)]
    logic_check(pattern)


def logic_check(pattern):
    """
    This module merely checks some logic and adds some con-specific strings
    """
    logic_one = pattern[:40]
    logic_two = str([NUM_TO_CHAR[value] for value in logic_one])
    logic_two = re.sub(r"\'\,\s\'", " ", str(logic_two))
    logic_two = re.sub(r"https\:\/\/t\.co\/\S+", "", str(logic_two))
    logic_two = re.sub(r"https\s\:\s", "", str(logic_two))
    logic_two = re.sub(r"\/\/t\.co\/\S+", "", str(logic_two))
    logic_two = re.sub(r"\[", "", str(logic_two))
    logic_two = re.sub(r"\]", "", str(logic_two))
    logic_two = re.sub(r"\@\s", "@", str(logic_two))
    logic_two = re.sub(r"\#\s", "#", str(logic_two))
    logic_two = re.sub(r"\\n", "", str(logic_two))
    logic_two = re.sub(r"\"", "", str(logic_two))
    logic_two = re.sub(r"\\", "", str(logic_two))
    logic_two = re.sub(r"\s\!", "!", str(logic_two))
    logic_two = re.sub(r"\(", "", str(logic_two))
    logic_two = re.sub(r"\)", "", str(logic_two))
    logic_two = re.sub(r"\s\:", ":", str(logic_two))
    logic_two = re.sub(r"https", "", str(logic_two))
    con = np.random.randint(0,7)
    if con == 0:
        api.PostUpdate(str(''.join([str(itemInLogic_two) for itemInLogic_two in logic_two]) + '#DECEPTICON @BlackHatEvents'))
        print(str(''.join([str(itemInLogic_two) for itemInLogic_two in logic_two]) + '#DECEPTICON @BlackHatEvents'))
        repeater()
    elif con == 1:
        api.PostUpdate(str(''.join([str(itemInLogic_two) for itemInLogic_two in logic_two]) + '#DECEPTICON @sectorca'))
        print(str(''.join([str(itemInLogic_two) for itemInLogic_two in logic_two]) + '#DECEPTICON @BlackHatEvents'))
        repeater()
    elif con == 2:
        api.PostUpdate(str(''.join([str(itemInLogic_two) for itemInLogic_two in logic_two]) + '#DECEPTICON @hackinthebox'))
        print(str(''.join([str(itemInLogic_two) for itemInLogic_two in logic_two]) + '#DECEPTICON @BlackHatEvents'))
        repeater()
    elif con == 3:
        api.PostUpdate(str(''.join([str(itemInLogic_two) for itemInLogic_two in logic_two]) + '#DECEPTICON @hackfest_ca'))
        print(str(''.join([str(itemInLogic_two) for itemInLogic_two in logic_two]) + '#DECEPTICON @BlackHatEvents'))
        repeater()
    elif con == 4:
        api.PostUpdate(str(''.join([str(itemInLogic_two) for itemInLogic_two in logic_two]) + '#DECEPTICON @hackfest_ca'))
        print(str(''.join([str(itemInLogic_two) for itemInLogic_two in logic_two]) + '#DECEPTICON @BlackHatEvents'))
        repeater()
    elif con == 5:
        api.PostUpdate(str(''.join([str(itemInLogic_two) for itemInLogic_two in logic_two]) + '#DECEPTICON @NoNameConOrg'))
        print(str(''.join([str(itemInLogic_two) for itemInLogic_two in logic_two]) + '#DECEPTICON @BlackHatEvents'))
        repeater()
    elif con == 6:
        api.PostUpdate(str(''.join([str(itemInLogic_two) for itemInLogic_two in logic_two]) + '#DECEPTICON @hopeconf'))
        print(str(''.join([str(itemInLogic_two) for itemInLogic_two in logic_two]) + '#DECEPTICON @BlackHatEvents'))
        repeater()
    else:
        api.PostUpdate(str(''.join([str(itemInLogic_two) for itemInLogic_two in logic_two]) + '#DECEPTICON @x33fcon'))
        print(str(''.join([str(itemInLogic_two) for itemInLogic_two in logic_two]) + '#DECEPTICON @BlackHatEvents'))
        repeater()


def repeater():
    """
    If a continuation is requested, this will trigger repetition.
    """
    os.remove(FILENAME)
    #sleeping_interval = POST_INTERVAL-(random.randint(0, 480))
    sleeping_interval = (random.randint(14400, 28800))
    print(f"\t\t\tSleeping for {sleeping_interval} seconds...\t\t\t")
    sleeping_hours = (sleeping_interval / 3600)
    print(f"\t\t\tSleeping for {sleeping_hours} hours...\t\t\t")
    time.sleep(sleeping_interval)
    subsequent()


def subsequent():
    """
    This initializes further repetition.
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
    hopper(api, USER_DF)


#USER = sys.argv[1]
#USER = USER.lower()
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
             tweet_mode='extended')
login()
