# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 21:08:59 2021

@author: HSingh
"""

#importing libraries
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
data = pd.read_csv('smsspamcollection/SMSSpamCollection',sep='\t',names=['label','Message'])


#Preprocessing and cleaning the dataset
import re
import nltk
from nltk.corpus import stopwords

#importing porter stemmer
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()

#importing lemmatizer
from nltk.stem import WordNetLemmatizer
lm=WordNetLemmatizer()

#initializing corpus
corpus_ps = []
corpus_lm = []

for i in range(len(data)):
    text=re.sub('[^a-zA-Z]',' ',data['Message'][i])     #removing non alphabetical characters
    text=text.lower()       #lowering the case
    text=text.split()       #splitting the words
    
    text_ps=[ps.stem(word) for word in text if word not in stopwords.words('english')]     #stemming
    text_lm=[lm.lemmatize(word) for word in text if word not in stopwords.words('english')]     #lemmatization
    
    text_ps=' '.join(text_ps)     #joining the words
    text_lm=' '.join(text_lm)
    
    corpus_ps.append(text_ps)     #appending the corpus
    corpus_lm.append(text_lm)

#Bag of Words Modeling
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000)     #Number of features are limited to 5000 to eliminate infrequent words

x_ps = cv.fit_transform(corpus_ps).toarray()
x_lm = cv.fit_transform(corpus_lm).toarray()

#preparing the dataset
y = pd.get_dummies(data['label'])
y = y.iloc[:,1].values

#Splitting the data into test and train
from sklearn.model_selection import train_test_split

x_tr1,x_ts1,y_tr1,y_ts1 = train_test_split(x_ps,y,test_size=0.2)   #porter stemmer
x_tr2,x_ts2,y_tr2,y_ts2 = train_test_split(x_lm,y,test_size=0.2)   #lemmatizer

#importing the Naive-Bayes Classifier
from sklearn.naive_bayes import MultinomialNB
model_ps = MultinomialNB()
model_lm = MultinomialNB()
model_ps.fit(x_tr1,y_tr1)
model_lm.fit(x_tr2,y_tr2)

#doing predictions
y_pred1 = model_ps.predict(x_ts1)
y_pred2 = model_lm.predict(x_ts2)

#performance evaluation
from sklearn.metrics import accuracy_score
print("Accuracy using Porter Stemmer:",accuracy_score(y_ts1,y_pred1))
print("Accuracy using Lemmatizer:", accuracy_score(y_ts2,y_pred2))

#Accuracy was obtained higher when porter stemmer is used
#Accuracy using Porter Stemmer: 98.4%
#Accuracy using Lemmatizer: 97.7%
from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(model_ps,x_ts1,y_ts1)
plot_confusion_matrix(model_lm,x_ts2,y_ts2)
plt.show()






