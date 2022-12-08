from data_loader import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

X_train = training_bs['lem_words']
y_train = training_bs['sentiment']
X_test = test['lem_words']
y_test = test['sentiment']

#X = df_training.lem_words
#y = df_training.sentiment
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

cvec = CountVectorizer()
cvec.fit(X_train)
X_train_cvec = cvec.transform(X_train)
X_test_cvec = cvec.transform(X_test)
tvec = TfidfVectorizer()
tvec.fit(X_train)
X_train_tvec = tvec.transform(X_train)
X_test_tvec = tvec.transform(X_test)
