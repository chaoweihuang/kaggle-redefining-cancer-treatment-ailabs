import os

import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn import metrics, model_selection


class Model:
    def __init__(self, **kwargs):
        params = {
            'eta': 0.02,
            'max_depth': 6,
            'min_child_weight': 1,
            'objective': 'multi:softprob',
            'eval_metric': 'mlogloss',
            'num_class': 9,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'silent': True
        }
        
        for key in kwargs:
            params[key] = kwargs[key]    

        self.params = params

    def cv(self, train, y, test, fold=5, repeat=3):
        preds = []

        print('Will do {fold}-fold cv for {repeat} times'.format(fold=fold, repeat=repeat))
        for shift in range(0, 100*repeat, 100):
            scores = []
            
            for i in range(fold):
                params = self.params
                params['seed'] = i + shift

                x1, x2, y1, y2 = model_selection.train_test_split(
                                    train, y, test_size=0.15,
                                    random_state=i+shift)
                
                watchlist = [(xgb.DMatrix(x1, y1), 'train'),
                             (xgb.DMatrix(x2, y2), 'valid')]

                model = xgb.train(
                            params, xgb.DMatrix(x1, y1), 1000,
                            watchlist, verbose_eval=50,
                            early_stopping_rounds=50)

                score = metrics.log_loss(
                                    y2, model.predict(xgb.DMatrix(x2)),
                                    labels = list(range(9)))

                print('\nlogloss for this fold:', score)
                scores.append(score)

                pred_test = model.predict(xgb.DMatrix(test))
                preds.append(pred_test)

            print('\nlogloss for this CV:', np.mean(scores))

        return np.asarray(preds)
