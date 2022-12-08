from data_loader import *
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import GridSearchCV, cross_val_score, cross_val_predict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from pprint import pprint
from time import time

def grid_search(p_cvec_lr, param_cvec_lr, X_train, y_train):
    try:
        grid_search = GridSearchCV(p_cvec_lr, param_cvec_lr, n_jobs=-1, verbose=1, cv=3)
        grid_search.fit(X_train, y_train)
        print("Best score: %0.3f" % grid_search.best_score_)
        print()
        #### get the best parameters
        best_parameters = grid_search.best_estimator_.get_params()
        print(best_parameters)
        for param_name in sorted(param_cvec_lr.keys()):
            print("\t%s: %r" % (param_name, best_parameters[param_name]))
    except Exception as e:
        print(e)

X_train = training_bs['lem_words']
y_train = training_bs['sentiment']


cvec = CountVectorizer()
cvec.fit(X_train)
X_train_cvec = cvec.transform(X_train)
tvec = TfidfVectorizer()
tvec.fit(X_train)
X_train_tvec = tvec.transform(X_train)
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

for i in range(2):
    k = mod_score['scores_cvec'].argsort()[-2:].iloc[i]
    print(k)
    if(k==1):
        p_cvec = Pipeline([
            ('vect', CountVectorizer()),
            ('clf', models[k])
            ])
        param_cvec = {'vect__max_df': (0.25, 0.5, 0.75, 1.0),
                         'vect__max_features': (None, 5000, 10000, 50000),
                         'vect__ngram_range': ((1, 1), (1, 2), (1,3), (1,4)),
                         'clf__max_depth':[None,1,2,3,4,5,6],
                         'clf__max_leaf_nodes':[5,6,7,8,9,10], 
                         'clf__min_samples_leaf':[1,2,3,4],
                        }
        #### Tfidf Vec
        p_tvec = Pipeline([
            ('vect', TfidfVectorizer()),
            ('clf', models[k])
            ])
        param_tvec = {'vect__max_df': (0.25, 0.5, 0.75, 1.0),
                         'vect__ngram_range': [(1, 1), (1, 2), (1, 3)],
                         'clf__max_depth':[None,1,2,3,4,5,6],
                         'clf__max_leaf_nodes':[5,6,7,8,9,10], 
                         'clf__min_samples_leaf':[1,2,3,4],
                        }
    elif(k==2):
        p_cvec = Pipeline([
            ('vect', CountVectorizer()),
            ('clf', models[k])
            ])
        param_cvec = {'vect__max_df': (0.25, 0.5, 0.75, 1.0),
                         'vect__max_features': (None, 5000, 10000, 50000),
                         'vect__ngram_range': ((1, 1), (1, 2), (1,3), (1,4)),
                         'clf__loss': ['log'],
                         'clf__penalty': ['l1','l2'],
                         'clf__alpha': np.logspace(-5,1,15),
                        }
        #### Tfidf Vec
        p_tvec = Pipeline([
            ('vect', TfidfVectorizer()),
            ('clf', models[k])
            ])
        param_tvec = {'vect__max_df': (0.25, 0.5, 0.75, 1.0),
                         'vect__ngram_range': [(1, 1), (1, 2), (1, 3)],
                         'clf__loss': ['log'],
                         'clf__penalty': ['l1','l2'],
                         'clf__alpha': np.logspace(-5,1,15),
                        }
    else:
        p_cvec = Pipeline([
            ('vect', CountVectorizer()),
            ('clf', models[k])
            ])
        param_cvec = {'vect__max_df': (0.25, 0.5, 0.75, 1.0),
                         'vect__max_features': (None, 5000, 10000, 50000),
                         'vect__ngram_range': ((1, 1), (1, 2), (1,3), (1,4)),
                         'clf__max_depth':[None,1,2,3,4,5,6],
                         'clf__loss': ['log'],
                         'clf__penalty': ['l1','l2'],
                         'clf__alpha': np.logspace(-5,1,15),
                         'clf__max_leaf_nodes':[8,9,10], 
                         'clf__min_samples_leaf':[1,2,3,4],
                        }
        #### Tfidf Vec
        p_tvec = Pipeline([
            ('vect', TfidfVectorizer()),
            ('clf', models[k])
            ])
        param_tvec = {'vect__max_df': (0.25, 0.5, 0.75, 1.0),
                         'vect__ngram_range': [(1, 1), (1, 2), (1, 3)],
                         'clf__max_depth':[None,1,2,3,4,5,6],
                         'clf__loss': ['log'],
                         'clf__penalty': ['l1','l2'],
                         'clf__alpha': np.logspace(-5,1,15),
                         'clf__max_leaf_nodes':[8,9,10], 
                         'clf__min_samples_leaf':[1,2,3,4],
                        }
    grid_search(p_tvec, param_tvec, X_train, y_train)
    print()
    grid_search(p_cvec, param_cvec, X_train, y_train)
    print()
