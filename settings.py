from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from sklearn.linear_model import RidgeClassifier
from xgboost import XGBClassifier

model = {
    "random-forest": RandomForestClassifier,
    "lightgbm": LGBMClassifier,
    "svm": SVC,
    "ridge": RidgeClassifier,
    "xgb": XGBClassifier
}
