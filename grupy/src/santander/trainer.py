import sklearn.ensemble as es
import sklearn.preprocessing as pp
from sklearn.grid_search import RandomizedSearchCV, GridSearchCV
import numpy as np


class Trainer(object):
    def __init__(self, model, data_to_train, features, target):
        self.model = model
        self.y = data_to_train[target].values
        self.features = features
        self.imp = pp.Imputer(strategy='median')
        self.impX = self.imp.fit_transform(data_to_train[features].values)

    def estimate_model(self, **kwargs_model):
        model_fit = self.model(**kwargs_model)
        model_fit = model_fit.fit(self.impX, self.y)

        return model_fit

    def find_best_params(self, method_search, **param_search):
        search = method_search(self.model(n_estimators=30), **param_search)

        return search.fit(self.impX, self.y).best_params_

    def predict(self, estimated_model, data_to_predict):
        X_to_predict = data_to_predict[self.features].values
        impX_to_predict = self.imp.transform(X_to_predict)

        return estimated_model.predict(impX_to_predict)
