import click
import mlflow
import yaml

with open("./parameters.yml") as f:
    parameters = yaml.safe_load(f)

parameters = parameters["mlflow"]
TRACKING_URI = parameters["TRACKING_URI"]
EXPERIMENT_NAME = parameters["EXPERIMENT_NAME"]
PROJECT_URI = parameters["PROJECT_URI"]

@click.command()
@click.option('--tracking_uri', default=TRACKING_URI)
@click.option('--experiment_name', default=EXPERIMENT_NAME)
@click.option('--project_uri', default=PROJECT_URI)
def main(tracking_uri, experiment_name, project_uri):
    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:  # 当該Experiment存在しないとき、新たに作成
        experiment_id = mlflow.create_experiment(name=experiment_name)
    else: # 当該Experiment存在するとき、IDを取得
        experiment_id = experiment.experiment_id

    run = mlflow.projects.run(uri=project_uri, env_manager="virtualenv", experiment_id=experiment_id)

if __name__ == "__main__":
    main()
