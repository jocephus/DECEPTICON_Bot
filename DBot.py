        print('\n\nUpdated Stats for all tweets:\n\n')
        ld2 = lexical_diversity(mainDF['Tweets'])
        print(f'\nThe Lexical Diversity of all Tweets is:\t\t\t\t\t{ld2}')
        ld3 = np.mean(mainDF['LD'])
        print(f'The Updated Statistical Lexical Diversity of all Tweets is:\t\t{ld3}')
        ld4 = np.std(mainDF['LD'])
        print(f'The Updated StdDev of Lexical Diversity of all Tweets is:\t\t{ld4}')
        timeStdDev = np.std(mainDF['Times'])
        print("\n\nTweets occur at this Updated interval:\t\n")
        postInterval = int(timeStdDev)
        print(f"\t{postInterval} seconds apart.\n\n")
        mainDF = mainDF.drop_duplicates()
        userDF = userDF.drop_duplicates()
        userDF.to_csv('user.csv')
        mainDF.to_csv('main.csv')
        shutil.move(os.path.join(directory, 'main.csv'), os.path.join(directory, 'main_repo', 'main.csv'))
        shutil.move(os.path.join(directory, user+'.csv'), os.path.join(directory, user, user+'.csv'))
        os.chdir(directory)
        #gonogo(api, directory, fSplit, eTime, ld, mainDF, userDF)
        repeater(api, directory, fSplit, eTime, ld, mainDF, userDF, postInterval)

def gonogo(api, directory, fSplit, eTime, ld, mainDF, userDF):
        gonogo = input("Continue? (Y/N)")
        if gonogo.lower() == 'y':
                print("Sleeping for 4 hours")
                time.sleep(60)
                subsequent(api, directory, fSplit, eTime, ld, mainDF, userDF)
        else:
                print("Goodbye")
                exit()

def repeater(api, directory, fSplit, eTime, ld, mainDF, userDF, postInterval):
        sleeping_interval = postInterval-(random.randint(0,480))
        print(f"\t\t\tSleeping for {sleeping_interval} seconds...\t\t\t")
        time.sleep(sleeping_interval)
        subsequent(api, directory, fSplit, eTime, ld, mainDF, userDF)

user = sys.argv[1]
user = user.lower()  
login() 
