# -*- coding: utf-8 -*-
"""
Classifier
"""
import os
import numpy as np
import lightgbm as lgb
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

FOLDER = "./data/hypotheses/embeds/"
TARGETS = ['LargeMediumNegativeES', 'SmallNegativeES', 'NullFinding',
           'SmallPositiveES', 'LargeMediumPositiveES']
X = np.load(os.path.join(FOLDER, "h_regular_es_d_x.npy"))
y = np.load(os.path.join(FOLDER, "h_regular_es_d_y.npy"))
y = y.reshape(y.shape[1])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=23)
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

params = {
    'objective': 'multiclass',
    'num_class': 5,
    'metric': 'multi_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9
}

bst = lgb.train(params, train_data, num_boost_round=100,
                valid_sets=[train_data, test_data],
                callbacks=[lgb.early_stopping(stopping_rounds=3),])

y_pred = bst.predict(X_test)
y_prex_max = y_pred.argmax(axis=1)

results = classification_report(y_test, y_prex_max, target_names=TARGETS)
print(results)