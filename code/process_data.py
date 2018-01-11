# -*- coding: utf-8 -*-
# Created by yhu on 2018/1/7.
# Describe:

import numpy as np
import pandas as pd

from scipy.stats import skew
from scipy.special import boxcox1p

from sklearn.preprocessing import RobustScaler

from . import settings


class ProcessData(object):

    def __init__(self):
        self.load_data()

    def load_data(self):
        train_raw = pd.read_csv(settings.TRAIN_FILE, index_col='id', encoding='gbk')
        test_raw = pd.read_csv(settings.TEST_FILE, index_col='id', encoding='gbk')

        new_columns = ['F{0}'.format(i + 1) for i in range(train_raw.shape[1])]
        new_columns[-1] = 'blood_sugar'
        feature_map = pd.Series(index=new_columns, data=train_raw.columns)
        train_raw.columns = new_columns
        test_raw.columns = new_columns[:-1]

        y_train = train_raw.blood_sugar
        all_feature_data = pd.concat([train_raw.iloc[:, :-1], test_raw], axis=0, join='outer').reset_index(drop=True)

        # 处理性别数据, other 是缺失值
        sex_dict = {u'男': 'male', u'女': 'female'}
        all_feature_data.F1 = all_feature_data.F1.apply(lambda x: sex_dict[x] if x in sex_dict.keys() else 'other')

        # 解析日期
        all_feature_data.F3 = all_feature_data.F3.apply(lambda x: pd.datetime.strptime(x, '%d/%m/%Y'))

        n_train = train_raw.shape[0]
        n_test = test_raw.shape[0]

        discrete_data = all_feature_data.iloc[:, :3]
        continuous_data = all_feature_data.iloc[:, 3:]

        # 缺失值比例太高, 删掉
        drop_feature_list = ['F19', 'F20', 'F21', 'F22', 'F23']
        continuous_data = continuous_data.drop(drop_feature_list, axis=1)

        # 用均值填充缺失值
        continuous_data = continuous_data.apply(lambda x: x.fillna(x.mean()))

        self.all_feature_data = all_feature_data
        self.continuous_data = continuous_data
        self.discrete_data = discrete_data
        self.y_train = y_train
        self.feature_map = feature_map
        self.n_train = n_train
        self.n_test = n_test

        return all_feature_data, y_train, feature_map, n_train, n_test

    def process_skewness(self):
        # 第一列是性别, 跳过处理
        skewed_feats = self.continuous_data.iloc[:, 1:].apply(lambda x: skew(x)).sort_values(ascending=False)
        skewness = pd.DataFrame({'Skew': skewed_feats})
        skewness = skewness[skewness.abs() > 0.75]  # 偏度大于0.75的需要处理

        lam = 0.15
        for feat in skewness.index:
            self.continuous_data[feat] = boxcox1p(self.continuous_data[feat], lam)
        return self.continuous_data

    def robust_scaler_data(self):
        rs = RobustScaler()
        self.continuous_data = rs.fit_transform(self.continuous_data)
        return self.continuous_data

    def process_date(self):
        self.discrete_data['day'] = self.discrete_data.F3.apply(lambda x: 'day_{0}'.format(x.day))
        self.discrete_data['month'] = self.discrete_data.F3.apply(lambda x: 'month_{0}'.format(x.month))
        self.discrete_data['weekday'] = self.discrete_data.F3.apply(lambda x: 'weekday_{0}'.format(x.dayofweek))
        self.discrete_data['weekofyear'] = self.discrete_data.F3.apply(lambda x: x.weekofyear)
        self.discrete_data['dayofyear'] = self.discrete_data.F3.apply(lambda x: x.dayofyear)

        # F3: 日期
        self.discrete_data = self.discrete_data.drop(['F3'], axis=1)

    def process_age(self):
        self.discrete_data['AGE'] = self.discrete_data.F2.apply(lambda x: 'AGE_{0}'.format(int(x / 10)))

    def process_one_hot(self):
        self.discrete_data = pd.get_dummies(self.discrete_data)
        self.discrete_data = self.discrete_data.values

    def process_data(self):
        self.y_train = np.log1p(self.y_train)

        self.process_skewness()
        self.robust_scaler_data()
        self.process_date()
        # self.process_age()
        self.process_one_hot()

        self.after_process_data = np.hstack([self.continuous_data, self.discrete_data])


def get_train_test_data():
    process_data = ProcessData()
    process_data.process_data()

    y_train = process_data.y_train
    after_process_data = process_data.after_process_data

    n_train = process_data.n_train
    n_test = process_data.n_test

    X_train = after_process_data[:n_train]
    X_test = after_process_data[n_train:]

    return X_train, X_test, y_train
