import os
from omegaconf import DictConfig
import hydra
from model_trainer import ModelTrainer

@hydra.main(version_base='1.3', config_path="conf", config_name="config.yaml")
def main(config: DictConfig):
    # MLModelTrainerのインスタンスを作成
    ml_model_trainer = ModelTrainer(config)
    ml_model_trainer.run()

if __name__ == "__main__":
    main()
