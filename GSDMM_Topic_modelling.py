#Load required python libraries
import re
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import string
import nltk
import warnings 
import pickle
from gsdmm import MovieGroupProcess
from nltk.stem.porter import *
import gensim.corpora as corpora
import pyLDAvis.gensim
import gensim
from gensim import corpora
from collections import Counter


# commands Retrieve Tweets from Twitter
#import tweepy as tw


# Define the search term and the date_since date as variables
#search_words = "#wildfires"
#date_since = "2018-11-16"


# Collect tweets
#tweets = tw.Cursor(api.search,q=search_words,lang="en",since=date_since).items(5)





# Read data
tweets = pd.read_csv('elonmusk_tweets.csv')

# Remove URL present in tweets
tweets['new'] = tweets['text'].apply(lambda x: re.sub(r"http\S+", "", x))


#  Removing Twitter Handles (@user)
def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)
        
    return input_txt    



# remove twitter handles (@user)
tweets['tidy_text'] = np.vectorize(remove_pattern)(tweets['new'], "@[\w]*")



# remove special characters, numbers, punctuations
tweets['tidy_text'] = tweets['tidy_text'].str.replace("[^a-zA-Z#]", " ")

 # Removing Short Words     
tweets['tidy_text'] = tweets['tidy_text'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))



# Tokenization
tweets['tok'] = tweets['tidy_text'].apply(lambda x: nltk.word_tokenize(x))


noun = []   
for  index, row in tweets.iterrows():
    noun.append([word for word,pos in nltk.pos_tag(row['tok']) if pos == 'NN' or pos =='NNP' or pos == 'NNS' or pos == 'NNPS'])
    
    
tweets['nouns'] = noun





tokenized_tweet = tweets['nouns']



# Stemming

stemmer = PorterStemmer()

tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x]) 



docs = tokenized_tweet.tolist()
vocab = set( x for doc in docs for x in doc)
n_terms = len(vocab)



#Set Hyperparameters
mgp = MovieGroupProcess(K=10, alpha=0.1, beta=0.1, n_iters=30)

#Train GSDMM model
y = mgp.fit(docs,n_terms)



doc_count = np.array(mgp.cluster_doc_count)






temp = mgp.cluster_word_distribution






# Topics sorted by the number of document they are allocated to
# Print results
top_index = doc_count.argsort()[-10:][::-1]
print('Most important Topics/clusters (by number of documents inside):', top_index)
print('*'*20)


for index in range(len(temp)):
    print('Topic :' ,index)
    j=temp[index]
    c=Counter(j)
    mc= c.most_common(5)
    print(mc)
    
    
    
    
    




