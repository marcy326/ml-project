import mlflow
from mlflow import MlflowClient
import yaml

with open("./parameters.yml") as f:
    parameters = yaml.safe_load(f)

parameters = parameters["mlflow"]
TRACKING_URI = parameters["TRACKING_URI"]
EXPERIMENT_NAME = parameters["EXPERIMENT_NAME"]
PROJECT_URI = parameters["PROJECT_URI"]

mlflow.set_tracking_uri(TRACKING_URI)
experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
if experiment is None:  # 当該Experiment存在しないとき、新たに作成
    experiment_id = mlflow.create_experiment(
                            name=EXPERIMENT_NAME)
else: # 当該Experiment存在するとき、IDを取得
    experiment_id = experiment.experiment_id

client = MlflowClient()


# mlflow.run(project_uri, env_manager="local", experiment_id=experiment_id)
run = mlflow.projects.run(uri=PROJECT_URI, env_manager="local", experiment_id=experiment_id)
