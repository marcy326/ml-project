import logging, os
import settings

import yaml
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.base import ClassifierMixin
import mlflow
import mlflow.data
from mlflow.models.signature import infer_signature

def load_yaml(file_path):
    with open(file_path) as f:
        yaml_file = yaml.safe_load(f)
    return yaml_file

def preprocess(df: pd.DataFrame, parameters: dict) -> pd.DataFrame:
    df_prep = df.copy()
    features = parameters["features"]
    target = parameters["target"]

    if target in df_prep:
        df_prep = df_prep[features + [target]]
    else:
        df_prep = df_prep[features]

    df_prep = pd.get_dummies(df_prep)

    return df_prep

def split_data(data: pd.DataFrame, parameters: dict) -> tuple:
    """Splits data into features and targets training and test sets.

    Args:
        data: Data containing features and target.
        parameters: Parameters defined in parameters/data_science.yml.
    Returns:
        Split data.
    """

    target = parameters["target"]
    X = data.drop(target, axis=1)
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=parameters["test_size"], random_state=parameters["random_state"]
    )
    return X_train, X_test, y_train, y_test

def parameter_tuning(X_train: pd.DataFrame, y_train: pd.Series, parameters: dict) -> dict:
    key = "param_grid"
    if key not in parameters or type(parameters[key])!=list or parameters[key]==[]:
        params = parameters
    return params


def train_model(X_train: pd.DataFrame, y_train: pd.Series, parameters: dict) -> ClassifierMixin:
    """Trains the linear regression model.

    Args:
        X_train: Training data of independent features.
        y_train: Training data for price.

    Returns:
        Trained model.
    """
    y_train = y_train.squeeze()
    ClfModel = settings.model[parameters["model_name"]]
    regressor = ClfModel(**parameters["model"])
    regressor.fit(X_train, y_train)
    return regressor

def model_predict(regressor: ClassifierMixin, data: pd.DataFrame, parameters: dict) -> pd.Series:
    """Prediction by trained model.

    Args:
        regressor: Trained model.
        data: Testing data of independent features.
    """
    target = parameters["target"]
    return pd.Series(regressor.predict(data), index=data.index, name=target)

def evaluate_model(y_pred: pd.Series, y_test: pd.Series) -> dict[str, float]:
    """Calculates and logs the coefficient of determination.

    Args:
        regressor: Trained model.
        X_test: Testing data of independent features.
        y_test: Testing data for price.
    """
    accuracy = accuracy_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    metrics = {"accuracy": float(accuracy), "mae": float(mae), "mse": float(mse), "rmse": float(rmse), "r2": float(r2)}
    logger = logging.getLogger(__name__)
    logger.info("Model has a accuracy of %.3f on test data.", accuracy)
    return metrics


def main():
    data = pd.read_csv("./data/train.csv", index_col=0)

    parameters = load_yaml("./parameters.yml")

    df_prep = preprocess(data, parameters)

    X_train, X_test, y_train, y_test = split_data(df_prep, parameters)

    clf = train_model(X_train, y_train, parameters)
    y_pred = model_predict(clf, X_test, parameters)
    metrics = evaluate_model(y_pred, y_test)

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        dataset = mlflow.data.from_pandas(df_prep, targets=parameters["target"])
        mlflow.log_input(dataset, context="training")
        
        # Log the parameters used for the model fit
        mlflow.log_params(parameters["model"])
        
        # Log the error metrics that were calculated during validation
        mlflow.log_metrics(metrics)
        signature = infer_signature(X_test, y_pred)
        mlflow.sklearn.log_model(clf, parameters["model_name"], signature=signature)

if __name__ == "__main__":
    main()
