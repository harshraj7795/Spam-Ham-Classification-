# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 21:08:59 2021

@author: HSingh
"""

#importing libraries
import pandas as pd

data = pd.read_csv('smsspamcollection/SMSSpamCollection',sep='\t',names=['label','Message'])


#Preprocessing and cleaning the dataset
import re
import nltk
from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()

#initializing corpus
corpus = []

for i in range(len(data)):
    text=re.sub('[^a-zA-Z]',' ',data['Message'][i])     #removing non alphabetical characters
    text=text.lower()       #lowering the case
    text=text.split()       #splitting the words
    text=[ps.stem(word) for word in text if word not in stopwords.words('english')]     #stemming
    text=' '.join(text)     #joining the words
    corpus.append(text)     #appending the corpus


#Bag of Words Modeling
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000)     #Number of features are limited to 5000 to eliminate infrequent words
x = cv.fit_transform(corpus).toarray()

#preparing the dataset
y = pd.get_dummies(data['label'])
y = y.iloc[:,1].values

#Splitting the data into test and train
from sklearn.model_selection import train_test_split
x_tr,x_ts,y_tr,y_ts = train_test_split(x,y,test_size=0.2)

#importing the Naive-Bayes Classifier
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(x_tr,y_tr)

y_pred = model.predict(x_ts)

#performance evaluation
from sklearn.metrics import classification_report
print(classification_report(y_ts,y_pred))

from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(model,x_ts,y_ts)






