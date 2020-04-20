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
        user_df = pd.DataFrame(columns=['User', 'Tweets', 'Times', 'LD'])
        #hopper(api, user_df)
        tokenization(api, user_df)
    else:
        user_df = pd.read_csv(os.path.join(DIRECTORY, 'files.csv'))
        #hopper(api, user_df)
        tokenization(api, user_df)


def lexical_diversity(text):
    return len(set(str(text))) / len(text)


def tokenization(api, user_df):
    results = api.GetUserTimeline(include_rts=False, count=200, exclude_replies=True)
    global boh
    global bom
    boh = []
    bom =[]
    for tweet in results:
        full_text = tweet.full_text
        full_split = str(full_text.split(' , '))
        hash_regex = re.compile(r"\s(?P<hashtag>\#\S+)\s")
        mention_regex = re.compile(r"\s(?P<mention>\@\S+)\s")
        links_regex = re.compile(r"\s(?P<links>https\:\/\/t\.co\/\S+)\s")
        links_bagger = links_regex.findall(str(full_split))
        mention_bagger = mention_regex.findall(str(full_split))
        hash_bagger = hash_regex.findall(str(full_split))
        if hash_bagger:
            boh.append(hash_bagger)
        if mention_bagger:
            bom.append(mention_bagger)
    hopper(api, user_df)


def hopper(api, user_df):
    results = api.GetUserTimeline(include_rts=False, count=200, exclude_replies=True)
    global bow
    global bow_freq
    global boh_freq
    global bom_freq
    bow = []
    for tweet in results:
        full_text = tweet.full_text
        tweet_time = tweet.created_at #Getting the UTC time
        make_time = time.mktime(time.strptime(tweet_time, "%a %b %d %H:%M:%S %z %Y"))
        epoch_time = int(make_time)
        lex_div = lexical_diversity(str(tweet))
        user_df = user_df.append({'User': USER, 'Tweets': full_text,
                                 'Times': epoch_time, 'LD': lex_div,},
                                   ignore_index=True, sort=True)
    sorter(full_text, user_df)
    user_df = user_df.drop_duplicates(subset=['Times'])
        
    
def sorter(full_text, user_df):    
    global inter
    inter = []
    for index, r in user_df.iterrows():
        w = r['Tweets']
        inter.append(w)
        tokenized_tweets = sent_tokenize(str(w))
        for w in tokenized_tweets:
            if w not in STOP_WORDS:
                w = w.lower()
                bow.append(w)
    user_df.to_csv('files.csv')
    pickle.dump(inter, open(DIRECTORY+'/nltk_data/full_text', 'wb'))
    for i in range(len(inter)):
        inter[i] = re.sub(r'\@\S+', '', full_text).split(' , ')
        inter[i] = re.sub(r'https\:\/\/t\.co\/\S+', '', full_text).split(' , ')
    stats(user_df)
    #   print('\n\nBOH Word Count:\n')
    boh_freq = freq_analysis(boh)
    #   print('\n\nBOM Word Count:\n')
    bom_freq = freq_analysis(bom)
    #   print('\n\nBOW Word Count:\n')
    bow_freq = freq_analysis(bow)
    generate()


def stats(user_df):
    global post_interval
    print(f"\n\nStats for {USER}'s tweets:\n\n")
    lex_div2 = lexical_diversity(user_df['Tweets'])
    print(f"\nThe Lexical Diversity of {USER}'s Tweets is:\t\t\t{lex_div2}")
    lex_div3 = np.mean(user_df['LD'])
    print(f"\nThe Statistical Mean Lexical Diversity of {USER}'s Tweets is:\t{lex_div3}")
    lex_div4 = np.std(user_df['LD'])
    print(f"\nThe StdDev of Lexical Diversity of {USER}'s Tweets is:\t\t{lex_div4}")
    time_stddev = np.std(user_df['Times'])
    print(f"\n\n{USER}'s Tweets occur at this interval:\t\n")
    post_interval = int(time_stddev)
    print(f"\n\t{post_interval} seconds apart.\n\n")
    print('\n')


def freq_analysis(subject):
    word_count = {}
    for data in subject:
        words = nltk.word_tokenize(str(data))
        for word in words:
            if word not in word_count.keys():
                word_count[word] = 1
            else:
                word_count[word] += 1
    

def generate():
    os.chdir(DIRECTORY+'/nltk_data')
    corpus = nltk.data.load('full_text', format='pickle')
    nltk.generate(corpus)
    
    
    
    
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
    os.makedirs('./files/nltk_data')
    DIRECTORY = DIRECTORY+'/files'
else:
    DIRECTORY = DIRECTORY+'/files'
STOP_WORDS = set(stopwords.words('english'))
login()
