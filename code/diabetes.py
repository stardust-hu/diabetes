# -*- coding: utf-8 -*-
# Created by yhu on 2018/1/7.
# Describe:

import numpy as np
import pandas as pd

from code import settings
from code.process_data import get_train_test_data
from code.model_sets import *
from code.utils import rmsle_cv, rmsle


train, test, y_train = get_train_test_data()


def test_single_model():
    rmsle_cv('Ridge', ridge, train, y_train, is_print_performance=True)

    rmsle_cv('ElasticNet', ENet, train, y_train, is_print_performance=True)

    rmsle_cv('Kernel Ridge', KRR, train, y_train, is_print_performance=True)

    rmsle_cv('Gradient Boosting', GBoost, train, y_train, is_print_performance=True)

    rmsle_cv('Xgboost', model_xgb, train, y_train, is_print_performance=True)

    rmsle_cv('LGBM', model_lgb, train, y_train, is_print_performance=True)


def test_staking():
    rmsle_cv('Stacking Averaged', stacked_averaged_models, train, y_train, is_print_performance=True)


def main():
    stacked_averaged_models.fit(train, y_train)
    stacked_train_pred = stacked_averaged_models.predict(train)
    stacked_pred = np.expm1(stacked_averaged_models.predict(test))
    print(rmsle(y_train, stacked_train_pred))

    ensemble = stacked_pred
    ensemble = list(map(lambda x: '%0.3f' % x, ensemble))
    sub = pd.DataFrame()
    sub['blood_sugar'] = ensemble

    submission_filename = settings.SUBMISSION_FILE.format("5th")
    sub.to_csv(submission_filename, index=False, header=None)


if __name__ == '__main__':
    # test_single_model()
    # test_staking()
    main()
