from feature_extraction import *

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier

models = [LogisticRegression(),
          RandomForestClassifier(),
          SGDClassifier(),
          SVC(),
          KNeighborsClassifier(), 
          MultinomialNB()
          ]

scores_cvec = []
scores_tvec = []
for model in models:
    print (model)
    score_cvec = cross_val_score(model, X_train_cvec, y_train, cv=3).mean()
    score_tvec = cross_val_score(model, X_train_tvec, y_train, cv=3).mean()
    print ('count vectoriser:', score_cvec)
    print ('tfidf vectoriser:', score_tvec)
    scores_cvec.append(score_cvec)
    scores_tvec.append(score_tvec)
    print ('_'*70)

mod = ['LR', 'RF', 'SGD', 'SVM', 'KNN', 'MultinomialNB']
mod_score = pd.DataFrame(zip(mod, scores_cvec, scores_tvec), columns = ['Model', 'scores_cvec', 'scores_tvec'])
print(mod_score)