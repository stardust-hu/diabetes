# -*- coding: utf-8 -*-
# Created by yhu on 2018/1/7.
# Describe:

from sklearn.model_selection import GridSearchCV

from . import utils
from .utils import rmsle, rmsle_cv
from .model_sets import model_lgb

from .process_data import get_train_test_data


train_data, test_data, y_train = get_train_test_data()


def grid_search():
    param_grid = {'num_leaves': [5, 6],
                  'learning_rate': [0.001, 0.01, 0.1],
                  'n_estimators': [500, 1000],
    }

    grid = GridSearchCV(model_lgb, param_grid, cv=10, scoring='neg_mean_squared_error', n_jobs=-1, verbose=0)
    grid.fit(train_data, y_train)

    model = grid.best_estimator_
    rmsle_cv('grid_search', model, train_data, y_train, is_print_performance=True)
    model.fit(train_data, y_train)
    y_pred = model.predict(train_data)
    rmsle(y_train, y_pred, is_log=True, is_print_performance=True)


def test():
    rmsle_cv('model_lgb', model_lgb, train_data, y_train, is_print_performance=True)
    model_lgb.fit(train_data, y_train)
    y_pred = model_lgb.predict(train_data)
    rmsle(y_train, y_pred, is_log=True, is_print_performance=True)


if __name__ == '__main__':
    test()
    grid_search()
