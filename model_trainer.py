# train.py
import logging
import yaml
from settings import ModelRegistry
import mlflow
import mlflow.data
from mlflow.models.signature import infer_signature
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV


class ModelTrainer:
    def __init__(self, config):
        self.config = config
        self.data = pd.read_csv(config.data.path, index_col=0)
        self.df_prep = self.preprocess_data()
        self.logger = logging.getLogger("mlflow.run")

    def preprocess_data(self):
        df_prep = self.data.copy()
        features = self.config.features
        target = self.config.target

        if target in df_prep:
            df_prep = df_prep[features + [target]]
        else:
            df_prep = df_prep[features]

        df_prep = pd.get_dummies(df_prep)

        return df_prep

    def split_data(self):
        target = self.config.target
        X = self.df_prep.drop(target, axis=1)
        y = self.df_prep[target]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config.test_size, random_state=self.config.random_state
        )
        return X_train, X_test, y_train, y_test

    def parameter_tuning(self, X_train, y_train):
        model_name = self.config.model.name
        ClfModel = ModelRegistry.models[model_name]

        param_grid = self.config.param_grid
        if not param_grid or not isinstance(param_grid, dict):
            return self.config.model.params

        grid_search = GridSearchCV(ClfModel(), param_grid, scoring='accuracy', cv=5, verbose=1)
        grid_search.fit(X_train, y_train)

        return grid_search.best_params_

    def train_model(self, X_train, y_train):
        tuned_params = self.parameter_tuning(X_train, y_train)
        model_name = self.config.model.name
        ClfModel = ModelRegistry.models[model_name]
        regressor = ClfModel(**tuned_params)
        regressor.fit(X_train, y_train)
        return regressor

    def model_predict(self, regressor, data):
        target = self.config.target
        return pd.Series(regressor.predict(data), index=data.index, name=target)

    def evaluate_model(self, y_pred, y_test):
        accuracy = accuracy_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        metrics = {"accuracy": float(accuracy), "mae": float(mae), "mse": float(mse), "rmse": float(rmse), "r2": float(r2)}
        self.logger.info("Model has an accuracy of %.3f on test data.", accuracy)
        return metrics

    def mlflow_logging(self):
        with mlflow.start_run() as run:
            run_id = run.info.run_id
            experiment_id = run.info.experiment_id

            # 生成したURLをログに出力
            mlflow_ui_url = f"http://mlflow.home/#/experiments/{experiment_id}/runs/{run_id}"
            self.logger.info(f"MLflow Run URL: {mlflow_ui_url}")

            dataset = mlflow.data.from_pandas(self.df_prep, targets=self.config.target)
            mlflow.log_input(dataset, context="training")

            mlflow.log_params(self.config.model.params)

            mlflow.log_metrics(self.metrics)
            signature = infer_signature(self.X_test, self.y_pred)
            mlflow.sklearn.log_model(self.clf, self.config.model.name, signature=signature)
            mlflow.log_artifact(__file__, "source")

    def run(self):
        self.X_train, self.X_test, self.y_train, self.y_test = self.split_data()
        self.clf = self.train_model(self.X_train, self.y_train)
        self.y_pred = self.model_predict(self.clf, self.X_test)
        self.metrics = self.evaluate_model(self.y_pred, self.y_test)
        self.mlflow_logging()
