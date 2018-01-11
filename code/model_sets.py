# -*- coding: utf-8 -*-
# Created by yhu on 2018/1/7.
# Describe:

from .utils import StackingAveragedModels

from sklearn.linear_model import ElasticNet, Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
import xgboost as xgb
import lightgbm as lgb


ridge = Ridge(alpha=2.08, random_state=11)

ENet = ElasticNet(alpha=0.0005, l1_ratio=0.5, random_state=2)

KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=0.25)

GBoost = GradientBoostingRegressor(n_estimators=500,
                                   learning_rate=0.05,
                                   max_depth=4,
                                   max_features='sqrt',
                                   min_samples_leaf=15,
                                   min_samples_split=10,
                                   loss='huber',
                                   random_state =3)

model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603,
                             gamma=0.0468,
                             learning_rate=0.05,
                             max_depth=3,
                             min_child_weight=1.7817,
                             n_estimators=2200,
                             reg_alpha=0.4640,
                             reg_lambda=0.8571,
                             subsample=0.5213,
                             silent=1,
                             seed=4,
                             nthread=-1)

model_lgb = lgb.LGBMRegressor(objective='regression',
                              num_leaves=5,
                              learning_rate=0.05,
                              n_estimators=720,
                              max_bin=55,
                              bagging_fraction=0.8,
                              bagging_freq=5,
                              feature_fraction=0.2319,
                              feature_fraction_seed=9,
                              bagging_seed=9,
                              min_data_in_leaf=6,
                              min_sum_hessian_in_leaf=11,
                              verbose=-1)

stacked_averaged_models = StackingAveragedModels(base_models=(model_xgb, model_lgb, ENet, GBoost, KRR),
                                                 meta_model=ridge)
