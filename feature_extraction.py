from data_loader import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

X_train = training_bs['lem_words']
y_train = training_bs['sentiment']


cvec = CountVectorizer()
cvec.fit(X_train)
X_train_cvec = cvec.transform(X_train)
tvec = TfidfVectorizer()
tvec.fit(X_train)
X_train_tvec = tvec.transform(X_train)