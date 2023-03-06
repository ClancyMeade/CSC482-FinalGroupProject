import snscrape.modules.twitter as sntwitter 

import string 
from textblob import TextBlob 
import pandas as pd

query = "lang:en AND Apple AND Amazon"
tweets = []
for i, tweet in enumerate(sntwitter.TwitterSearchScraper(query).get_items()):
    if i > 1000: 
        break 
    else: 
        tweets.append([tweet.date, tweet.content])
       #tweets.append([tweet.date, tweet.id, tweet.url, tweet.user.username, tweet.sourceLabel, tweet.user.location, tweet.content, tweet.likeCount, tweet.retweetCount])
#df = pd.DataFrame(tweets, columns = ['Date', 'ID', 'url', 'username', 'source', 'location', 'tweet', 'num_of_likes', 'num_of_retweet'])
df = pd.DataFrame(tweets, columns = ['Date', 'Content'])
df.to_csv('sentiment.csv', mode = 'a')