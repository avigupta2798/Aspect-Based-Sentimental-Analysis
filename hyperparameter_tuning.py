from sklearn.model_selection import GridSearchCV, cross_val_score, cross_val_predict
from model import *

from sklearn.pipeline import Pipeline


def grid_search(p_cvec_lr, param_cvec_lr, X_train, y_train):
    grid_list=[]
    try:
        grid_search = GridSearchCV(p_cvec_lr['clf'], param_cvec_lr, n_jobs=-1, verbose=1, cv=3)
        grid_search.fit(X_train, y_train)
        print("Best score: %0.3f" % grid_search.best_score_)
        print()
        #### get the best parameters
        best_parameters = grid_search.best_estimator_.get_params()
        best_estimator = grid_search.best_estimator_
        print(best_parameters)
        for param_name in sorted(param_cvec_lr.keys()):
            print("\t%s: %r" % (param_name, best_parameters[param_name]))
    except Exception as e:
        print(e)
    grid_list = [best_parameters, best_estimator]
    return grid_list
new_tvec={}
new_cvec={}
for i in range(2):
    k = mod_score['scores_cvec'].argsort()[-2:].iloc[i]
    print(k)
    if(k==1):
        p_cvec = Pipeline([
            ('vect', CountVectorizer()),
            ('clf', models[k])
            ])
        param_cvec = {'max_depth':[None,1,2,3,4,5,6],
                         'max_leaf_nodes':[8,9,10], 
                         'min_samples_leaf':[1,2,3,4],
                        }
        #### Tfidf Vec
        p_tvec = Pipeline([
            ('vect', TfidfVectorizer()),
            ('clf', models[k])
            ])
        param_tvec = {'max_depth':[None,1,2,3,4,5,6],
                         'max_leaf_nodes':[8,9,10], 
                         'min_samples_leaf':[1,2,3,4],
                        }
    elif(k==2):
        p_cvec = Pipeline([
            ('vect', CountVectorizer()),
            ('clf', models[k])
            ])
        param_cvec = {'loss': ['log'],
                         'penalty': ['l1','l2'],
                         'alpha': np.logspace(-5,1,15),
                        }
        #### Tfidf Vec
        p_tvec = Pipeline([
            ('vect', TfidfVectorizer()),
            ('clf', models[k])
            ])
        param_tvec = {'loss': ['log'],
                         'penalty': ['l1','l2'],
                         'alpha': np.logspace(-5,1,15),
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
    tvec = grid_search(p_tvec, param_tvec, X_train_tvec, y_train)
    print()
    cvec = grid_search(p_cvec, param_cvec, X_train_cvec, y_train)
    print()
    new_tvec[k] = tvec
    new_cvec[k] = cvec
