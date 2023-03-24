import snscrape.modules.twitter as sntwitter 
import random
import string 
from textblob import TextBlob 
import pandas as pd

def get_recent_tweets(brandName, count):
    query = "lang:en "+brandName+ " I"
    tweets = []
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(query).get_items()):
        if i >= count: 
            break 
        else: 
            tweets.append([tweet.content])
    return tweets
    df = pd.DataFrame(tweets, columns = ['Content'])

def fetchData(brandName): #fetch 100 most recent tweets with brand name and I in it 
    query = "lang:en "+brandName+ " I"
    tweets = []
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(query).get_items()):
        if i >= 100: 
            break 
        else: 
            tweets.append([tweet.content])
    df = pd.DataFrame(tweets, columns = ['Content'])
    df.to_csv(brandName+".txt", mode = 'a', header=False, index=False) #can comment out index=False 

def randomize_training(brandName): # run this ONCE to separate training and testing files
    f = open(brandName+".txt", 'r')
    temp = f.read().split("</L>")
    training = open("./data/" + brandName+"_training.txt", 'a')
    testing = open(brandName+"_testing.txt", 'a')
    random.shuffle(temp)
    for i in range(0,len(temp)):
        line = temp[i]
        if len(line) <= 2: 
            continue
        s= line+"</L>"
        if i <=80:
            training.write(s)
        else:
            testing.write(s)
    
def get_training_data(brandName): # returns training data in [tweet, +/0/-] format
    f = open("./data/"+brandName+"_training.txt", 'r')
    temp = f.read().split("</L>")
    text = []
    for t in temp:
        split = t.split("<L>")
        if len(split[0]) < 2:
            continue
        split[0] = split[0].replace("\n", " ")
        text.append(split)

    return text # as [text, '+/0/-']

def get_testing_data(brandName): #
    f = open("./data/"+brandName+"_testing.txt", 'r')
    temp = f.read().split("</L>")
    text = []
    for t in temp:
        split = t.split("<L>")
        if len(split[0]) < 2:
            continue
        split[0] = split[0].replace("\n", " ").strip()
        text.append(split)

    return text # as [text, '+/0/-']
