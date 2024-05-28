# -*- coding: utf-8 -*-
"""
Classifier
"""
import os
import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, accuracy_score, log_loss, \
    precision_score, recall_score, f1_score


class HypothesesClassifier:
    """ Main class for framing hypothesis generation as a classification task """
    def __init__(self, data_path, X_path, y_path):
        """ Init model and data """
        self.data = pd.read_csv(data_path, index_col=0)
        self.X = np.load(X_path)
        self.y = np.load(y_path)
        self.y = self.y.reshape(self.y.shape[1])

        self.model = LGBMClassifier(
            objective='multiclass', n_jobs=-1, random_state=23,
            force_col_wise=True,
            verbosity=1,)

    def init_model(self, **params):
        config = {
            "objective": "multicalss", "n_jobs": -1, "random_state": 23,
            "force_col_wise": True, "verbosity": 1,
            "learning_rate": params.get("learning_rate", )
        }
        for key, default_val in [
            ("learning_rate", 0.1), ("n_estimators", 100),
            ("max_depth", -1), ("num_leaves", 31), ("colsample_bytree", 1.0)
        ]:
            config.update({key: parals.get(key, default_val)})
        self.model = LGBMClassifier(**config)
    
    def train_test_split(self, test_size: float = 0.2, random_state: int = 23):
        """ Random train/test split, with indexes to trace back original data """
        indices = np.arange(self.X.shape[0])
        X_train, X_test, y_train, y_test, train_indices, test_indices = \
            train_test_split(self.X, self.y, indices, test_size=test_size,                  
                             random_state=random_state)
        return X_train, X_test, y_train, y_test, train_indices, test_indices
    
    def run_grid_search(self, param_grid, X_train, y_train, cv: int = 5):
        """ Run param grid for grid search """
        grid_search = GridSearchCV(
            estimator=self.model, param_grid=param_grid, cv=cv
        )
        grid_search.fit(X_train, y_train)
        print('Best score:', grid_search.best_score_)
        model = grid_search.best_estimator_
        print('Best params:', model.get_params())
        return grid_search, model
    
    @staticmethod
    def get_metrics(y_predicted, y_true):
        """ Compute metrics: R2, MAE, RMSE """
        # mae = mean_absolute_error(y_true, y_predicted)
        # r2score = r2_score(y_true, y_predicted)
        # mse = mean_squared_error(y_true, y_predicted)
        # return {"mae": mae, "r2": r2_score, "mse": mse}
        results = {"accuracy": accuracy_score(y_true, y_predicted)} 
        # precision = precision_score(y_true, y_predicted, average=None)
        # recall = recall_score(y_true, y_predicted, average=None)
        # f1 = f1_score(y_true, y_predicted, average=None)

        for avg in ['macro', 'micro', 'weighted']:
            for (metric, label) in [
                (precision_score, "precision"), (recall_score, "recall"),
                (f1_score, "f1")]:
                results.update({label: metric(y_true, y_predicted, average=avg)})


        return results

    
    def __call__(self, params):
        X_train, X_test, y_train, y_test, train_indices, test_indices = \
            self.train_test_split()
        self.init_model(**params)
        self.model.fit(X_train, y_train)
        # to be continued


if __name__ == '__main__':
    FILENAME = "h_regular_es_r"
    DATA_PATH = f"./data/hypotheses/entry/{FILENAME}.csv"
    FOLDER = "./data/hypotheses/embeds/"
    X_PATH = os.path.join(FOLDER, f"{FILENAME}_x.npy")
    Y_PATH = os.path.join(FOLDER, f"{FILENAME}_y.npy")
    HC = HypothesesClassifier(
        data_path=DATA_PATH, X_path=X_PATH, y_path=Y_PATH)
    
    X_train, X_test, y_train, y_test, train_indices, test_indices = \
        HC.train_test_split()
    # PARAM_GRID = {
    #     # 'learning_rate':[0.01],
    #     # 'n_estimators':[10000, 1000],
    #     'n_estimators': [100],
    #     # 'max_depth':[6, -1],
    #     # 'num_leaves':[40, 60, 65, 70],#, 128, 256, 512],
    #     # 'colsample_bytree':[0.7, 1],
    # }
    # CV = 5
    # GS, MODEL = HC.run_grid_search(param_grid=PARAM_GRID, X_train=X_train, y_train=y_train, cv=CV)
    MODEL = HC.model
    MODEL.fit(X_train, y_train)

    y_pred_train = MODEL.predict(X_train)
    y_pred_test = MODEL.predict(X_test)
    print(y_pred_train.shape)
    print(y_pred_test.shape)
    print(y_train.shape)
    print(y_test.shape)
    print(HC.get_metrics(y_predicted=y_pred_train, y_true=y_train))
    print(HC.get_metrics(y_predicted=y_pred_test, y_true=y_test))




