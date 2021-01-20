import pandas as pd
from textblob import TextBlob
#importing the data

ds=pd.read_csv("CEH_exam_negative_reviews.csv")
ds

#converting the csv file to string format

dataset=ds.to_string(index=False)
type(dataset)
dataset

blob = TextBlob(dataset)
print(blob.sentiment)

#data cleaning

import re
dataset = re.sub("[^A-Za-z0-9]+"," ",dataset)

#data tokenizing

import nltk
from nltk.tokenize import word_tokenize
Tokens = word_tokenize(dataset)
print(Tokens)

from nltk.probability import FreqDist
fdist = FreqDist()

for word in Tokens:
    fdist[word]+=1
fdist
fdist.plot(20)

#stemming

from nltk.stem import PorterStemmer
pst=PorterStemmer()
pst.stem("having")

# removing stopwords

import nltk.corpus
stopwords = nltk.corpus.stopwords.words("english")
stopwords[0:10]

# getting rid of stopwords

filtered_sentence = []   
for FinalWord in Tokens:
    if FinalWord not in stopwords:
        filtered_sentence.append(FinalWord)  

print(filtered_sentence)  
len(filtered_sentence)
len(Tokens)

#calculating final sentimental score
filtered_sentence = ' '.join([str(elem) for elem in filtered_sentence]) 
print(filtered_sentence)

score=TextBlob(filtered_sentence)
print(score.sentiment)

####--------------score is 0.329636 -------------------

from wordcloud import WordCloud
word_cloud = WordCloud(width=512,height=512,background_color="white",stopwords=stopwords).generate(filtered_sentence)
import matplotlib.pyplot as plt
plt.imshow(word_cloud)




































