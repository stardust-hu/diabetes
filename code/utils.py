# -*- coding: utf-8 -*-
# Created by yhu on 2018/1/7.
# Describe:

import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error


def rmsle_cv(model_name, model, X_train, y_train, n_folds=5, is_print_performance=False):
    kf = KFold(n_folds, shuffle=True, random_state=0).get_n_splits(X_train)
    rmse = np.sqrt(-cross_val_score(model, X_train, y_train, scoring='neg_mean_squared_error', cv=kf))

    if is_print_performance is True:
        print("\n{} score: {:.4f} ({:.4f})\n".format(model_name, rmse.mean(), rmse.std()))
    return rmse


def rmsle(y_true, y_pred, is_log=True, is_print_performance=False):
    if is_log is True:
        y_true = np.expm1(y_true)
        y_pred = np.expm1(y_pred)
    score = np.sqrt(mean_squared_error(y_true, y_pred))
    if is_print_performance is True:
        print("\nscore: {:.4f}\n".format(score))
    return


class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models

    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]
        for model in self.models_:
            model.fit(X, y)
        return self

    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        return np.mean(predictions, axis=1)


class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds

    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)

        k_fold = KFold(n_splits=self.n_folds, shuffle=True, random_state=0)

        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, hold_out_index in k_fold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_index], y[train_index])
                y_pred = instance.predict(X[hold_out_index])
                out_of_fold_predictions[hold_out_index, i] = y_pred
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self

    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([
                model.predict(X)
                for model in base_models]).mean(axis=1)
            for base_models in self.base_models_])
        return self.meta_model_.predict(meta_features)


if __name__ == '__main__':
    pass
