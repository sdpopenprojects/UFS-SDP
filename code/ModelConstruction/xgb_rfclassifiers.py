# coding=utf-8
import warnings
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

def random_forest_classifier(train_X, train_y):
    """
    Random Forest model
    :param train_X: features
    :param train_y: labels
    :return: Random Forest model
    """
    model = RandomForestClassifier()
    model.fit(train_X, train_y)
    return model

def xgboost(training_data_x, training_data_y):
    """
    XGBoost model
    :param training_data_x: features
    :param training_data_y: labels
    :return: XGBoost model
    """
    model = XGBClassifier()
    model.fit(training_data_x, training_data_y)
    return model
