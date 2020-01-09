	
def search_space(model):
    '''
    Defines a search space for Bayeasian Hyperparamter optimization.
    Currently, the following models are supported:
        KNeighborsClassifier
        SVC
        LogisticRegression
        RandomForestClassifier
        XGBClassifier
    '''
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from hyperopt import hp
    import numpy as np
    from hyperopt.pyll import scope
    from xgboost import XGBClassifier

        
    if model not in [RandomForestClassifier, SVC, LogisticRegression, KNeighborsClassifier, XGBClassifier]:
        print(f'{model} is not supported')
        return
        
    #model = model.lower()
    space = {}
 
    if model == KNeighborsClassifier:
        space = {
            'n_neighbors': hp.choice('n_neighbors', range(1,100)),
            'scale': hp.choice('scale', [0, 1]),
            'normalize': hp.choice('normalize', [0, 1]),
            'p': scope.int(hp.uniform('p', 1, 10)),
            'cv': hp.choice('cv', ['btscv', 'tscv', 'cv']),
            'n_jobs': -1,
        }
 
    elif model == SVC:
         space = {
             #'kernel': hp.choice('kernel', ['linear', 'sigmoid', 'poly', 'rbf']),
             'scale': hp.choice('scale', [0, 1]),
             'normalize': hp.choice('normalize', [0, 1]),
             'gamma': hp.loguniform('gamma', np.log(1E-15), np.log(100)),
             'C': hp.loguniform('C', np.log(1), np.log(1E6)),
             'cv': hp.choice('cv', ['btscv', 'tscv', 'cv']),
             'model': SVC,
         }
 
    elif model == LogisticRegression:
         space = {
             'warm_start' : hp.choice('warm_start', [True, False]), 
             'fit_intercept' : hp.choice('fit_intercept', [True, False]),
             'tol' : hp.uniform('tol', 0.00001, 0.0001),
             'C' : hp.uniform('C', 0.05, 3),
             'solver' : hp.choice('solver', ['newton-cg', 'lbfgs', 'liblinear']),
             'max_iter' : hp.choice('max_iter', range(100,1000)),
             'scale': hp.choice('scale', [0, 1]),
             'normalize': hp.choice('normalize', [0, 1]),
             'multi_class' : 'auto',
             'class_weight' : 'balanced',
             n_jobs: -1,
      }
    elif model == RandomForestClassifier:
       space = {
           'max_depth': scope.int(hp.uniform('max_depth', 1, 300)),
           'max_features': hp.choice('max_features', ['auto', scope.int(hp.uniform('num', 1, 10))]),
           'n_estimators': scope.int(hp.uniform('n_estimators', 2, 500)),           
           'criterion': hp.choice('criterion', ['entropy', 'gini']),
           'cv': hp.choice('cv', ['btscv', 'tscv', 'cv']),
           'min_samples_leaf': scope.int(hp.uniform('min_samples_leaf', 1, 20)),
           'bootstrap': hp.choice('bootstrap', [False]),
           'n_jobs': -1,
     }
    
    elif model == XGBClassifier:
       space = {
           'booster': hp.choice('booster', ['gbtree', 'gblinear', 'dart']),
           'max_depth': scope.int(hp.uniform('max_depth', 1, 300)),
           'n_estimators': scope.int(hp.uniform('n_estimators', 2, 1000)),          
           'learning_rate': hp.loguniform('learning_rate', np.log(1E-3), np.log(100)),
           'subsample': 1,
           'colsample_bytree': hp.uniform('colsample_bytree', 0, 1),
           'gamma': hp.choice('gamma', [0, 1, 5, 10]),
           #'cv': hp.choice('cv', ['btscv', 'tscv', 'cv']),
           'cv': hp.choice('cv', ['btscv']),

           'n_jobs': -1,
     }
        
    
    space['model'] = model
    
    return space

from atomm.Methods import BlockingTimeSeriesSplit
from sklearn.model_selection import TimeSeriesSplit, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import time
from hyperopt import space_eval
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.metrics import accuracy_score
from sklearn.ensemble import BaggingClassifier
from sklearn.multiclass import OneVsRestClassifier
import pandas as pd
import numpy as np
from sklearn.svm import SVC

def BayesianSearch(
        param_space,
        model, 
        X_train,
        y_train, 
        X_test,
        y_test, 
        num_eval, 
        silent=False
    ):

    start = time.time()
    
    def cross_val_score_mod(clf,            
                            X_, 
                            y_,
                            cv,
          ):
        score = []
        idx = pd.IndexSlice
        for train, test in cv:
            clf.fit(X_.iloc[idx[train], :], y_.iloc[idx[train]])
            pred = clf.predict(X_.iloc[idx[test], :])
            score_ = accuracy_score(y_.iloc[idx[test]], pred)
            score.append(score_)
        return np.array(score)
    
    def scale_normalize(params, data):
        s = params.get('scale')
        n = params.get('normalize')
        if s:
            data_ = StandardScaler().fit_transform(data)
            data = pd.DataFrame(data=data_, columns=data.columns, index=data.index)
        if n:
            data_ = MinMaxScaler().fit_transform(data)
            data = pd.DataFrame(data=data_, columns=data.columns, index=data.index)

        return data
    
    def get_cv_method(params, X_):
        n_splits = 5
        cv = params.get('cv')
        if cv == 'tscv':
            cv = TimeSeriesSplit(n_splits=n_splits).split(X_)
        elif cv == 'btscv':
            cv = BlockingTimeSeriesSplit(n_splits=n_splits).split(X_)
        elif cv == 'pcv':
            cv = PurgedKFold()
        else:
            cv = KFold(n_splits=n_splits).split(X_)
        return cv

    
    def objective_function(params):
        model_fd = params.get('model')
        if model_fd != None:
            model = model_fd
        X_ = X_train
        X_ = scale_normalize(params, X_train)
        cv = get_cv_method(params, X_)
        try: del params['cv']
        except: pass
        try: del params['model']
        except: pass
        try: del params['scale']
        except: pass
        try: del params['normalize']
        except: pass
        clf = model(**params)
        if model == SVC:
            n_estimators = 15
            clf = OneVsRestClassifier(BaggingClassifier(SVC(**params), max_samples=1.0 / n_estimators, n_estimators=n_estimators, n_jobs=-1))
        score = cross_val_score_mod(
            clf, 
            X_, 
            y_train,
            cv=cv,
        ).mean()
        return {'loss': -score, 'status': STATUS_OK}

    trials = Trials()
    best_param = fmin(
        fn=objective_function, 
        space=param_space, 
        algo=tpe.suggest, 
        max_evals=num_eval, 
        trials=trials,
        #rstate=np.random.RandomState(1)
    )
    loss = [x['result']['loss'] for x in trials.trials]
    results = space_eval(param_space, best_param)
    best_param = space_eval(param_space, best_param)
    try: del best_param['cv']
    except: pass
    try: del best_param['model']
    except: pass
    try: del best_param['scale']
    except: pass
    try: del best_param['normalize']
    except: pass
    clf_best = model(
            **best_param
    )
    clf_best.fit(X_train, y_train)
    clf_best.fit(X_train, y_train)
    test_score = clf_best.score(X_test, y_test)
    if not silent:
        print('##### Results #####')
        print('Score best parameters: ', min(loss))
        print('Best parameters: ', results)
        print('Test Score: ', test_score)
        print('Parameter combinations evaluated: ', num_eval)
        print('Time elapsed: ', time.time() - start)
    return test_score, clf_best, best_param
