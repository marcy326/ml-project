from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from sklearn.linear_model import RidgeClassifier
from xgboost import XGBClassifier

class ModelRegistry:
    models = {}

def register_model(name, model_class):
    ModelRegistry.models[name] = model_class

register_model("random-forest", RandomForestClassifier)
register_model("lightgbm", LGBMClassifier)
register_model("svm", SVC)
register_model("ridge", RidgeClassifier)
register_model("xgb", XGBClassifier)
