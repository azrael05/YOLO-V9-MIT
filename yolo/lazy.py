import sys
import datetime
from pathlib import Path

import hydra
from lightning import Trainer

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from yolo.config.config import Config
from yolo.tools.solver import InferenceModel, TrainModel, ValidateModel
from yolo.tools.solver_cls import TrainModel_CLS, InferenceModel_CLS, ValidateModel_CLS
from yolo.tools.solver import TrainModel, InferenceModel, ValidateModel
from yolo.utils.logging_utils import setup
import logging
logging.basicConfig(filename=r"logs\timer.txt", filemode="a")
logger = logging.getLogger("timer")

@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: Config):
    callbacks, loggers, save_path = setup(cfg)
    callbacks = []
    print(cfg)
    if "cls" in cfg.model.name:
        cfg.weight = ""
    
    start = datetime.datetime.now()
    trainer = Trainer(
        accelerator="auto",
        max_epochs=getattr(cfg.task, "epoch", None),
        precision="32",
        callbacks=callbacks,
        logger=loggers,
        log_every_n_steps=1,
        gradient_clip_val=10,
        gradient_clip_algorithm="value",
        deterministic=True,
        # enable_progress_bar=not getattr(cfg, "quite", False),
        default_root_dir=save_path,
    )
    time_taken = datetime.datetime.now() - start
    logger.critical(f"Time for loading trainer {time_taken}")
    if "cls" in cfg.model.name:
        if cfg.task.task == "train":
            start = datetime.datetime.now()
            model = TrainModel_CLS(cfg)
            time_taken = datetime.datetime.now() - start
            logger.critical(f"Time taken for model building {time_taken}")
            trainer.fit(model)
            # model.train()
        if cfg.task.task == "validation":
            model = ValidateModel_CLS(cfg)
            trainer.validate(model)
        if cfg.task.task == "inference":
            model = InferenceModel_CLS(cfg)
            trainer.predict(model)
    else:
        if cfg.task.task == "train":
            model = TrainModel(cfg)
            trainer.fit(model)
        if cfg.task.task == "validation":
            model = ValidateModel(cfg)
            trainer.validate(model)
        if cfg.task.task == "inference":
            model = InferenceModel(cfg)
            trainer.predict(model)


if __name__ == "__main__":
    main()
