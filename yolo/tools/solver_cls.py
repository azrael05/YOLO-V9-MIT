from math import ceil
from pathlib import Path

from lightning import LightningModule
from torchmetrics.detection import MeanAveragePrecision
from sklearn.metrics import accuracy_score
import numpy as np
from tqdm import tqdm
import torch
from sklearn.metrics import classification_report

from yolo.config.config import Config
from yolo.model.yolo import create_model
from yolo.model.yolo_cls import create_model_cls
from yolo.tools.data_loader_cls import create_dataloader_cls
from yolo.tools.drawer import draw_bboxes
from yolo.tools.loss_functions import  create_loss_function_cls
from yolo.utils.bounding_box_utils import create_converter, to_metrics_format
from yolo.utils.model_utils import PostProcess, create_optimizer, create_scheduler
import datetime
from yolo.tools.solver import BaseModel
import logging
logger = logging.getLogger("timer")
logging.basicConfig(filename=r"logs\timer.txt", filemode="a")

class ValidateModel_CLS(BaseModel):
    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg
        if self.cfg.task.task == "validation":
            self.validation_cfg = self.cfg.task
        else:
            self.validation_cfg = self.cfg.task.validation
        self.val_loader = create_dataloader_cls(self.validation_cfg.data, self.cfg.dataset, self.validation_cfg.task)
        self.ema = self.model
        # self.actual = []
        # self.predicted = []
        # self.length = len(self.val_loader)

    def val_dataloader(self):
        return self.val_loader

    def validation_step(self, batch, batch_idx):
        # if len(self.predicted) == self.length:
        #     self.predicted = []
        #     self.actual = []
        images, targets = batch
        H, W = images.shape[2:]
        predicts = self.ema(images)
        # self.metric.update(
        #     [to_metrics_format(predict) for predict in predicts], [to_metrics_format(target) for target in targets]
        # )
        # class_preds = torch.argmax(predicts["Main"], dim=1)
        # for class_pred in class_preds:
        #     self.predicted.append(class_pred)
        # for target in targets:
        #     self.actual.append(target)
        return predicts

    def on_validation_epoch_end(self):
        # epoch_metrics = self.metric.compute()
        # del epoch_metrics["classes"]
        # self.log_dict(epoch_metrics, prog_bar=True, sync_dist=True, rank_zero_only=True)
        # self.log_dict(
        #     {"PyCOCO/AP @ .5:.95": epoch_metrics["map"], "PyCOCO/AP @ .5": epoch_metrics["map_50"]},
        #     sync_dist=True,
        #     rank_zero_only=True,
        # )
        # self.metric.reset()
        # print(classification_report(self.actual, self.predicted))
        pass


class TrainModel_CLS(ValidateModel_CLS):
    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg
        start = datetime.datetime.now()
        self.train_loader = create_dataloader_cls(self.cfg.task.data, self.cfg.dataset, self.cfg.task.task)
        time_taken = datetime.datetime.now() - start
        logger.critical(f"Dataloader creation time{time_taken}")
        print(len(self.train_loader))
        self.aactual = []
        self.ppredicted = []
        self.full_length = len(self.train_loader)

    def train_dataloader(self):
        return self.train_loader
    
    def setup(self, stage):
        # super().setup(stage)
        self.loss_fn = create_loss_function_cls(self.cfg)

    def on_train_epoch_start(self):
        if self.aactual != []:
            print(self.aactual)
            print(self.ppredicted)
            print(classification_report(self.aactual, self.ppredicted))
            self.aactual = []
            self.ppredicted = []
        self.trainer.optimizers[0].next_epoch(
            ceil(len(self.train_loader) / self.trainer.world_size), self.current_epoch
        )
        self.aactual = []
        self.ppredicted = []

    def training_step(self, batch, batch_idx):
        lr_dict = self.trainer.optimizers[0].next_batch()
        images, targets= batch
        predicts = self(images)
        batch_size = len(images)
        loss = self.loss_fn(predicts["Main"], targets)
        cclass_preds = torch.argmax(predicts["Main"], dim=1)
        for cclass_pred in cclass_preds.detach().numpy():
            self.ppredicted.append(cclass_pred)
        for target in targets.detach().numpy():
            self.aactual.append(np.argmax(target))
        print(len(self.aactual), len(self.ppredicted))
        # self.log_dict(lr_dict, prog_bar=False, logger=True, on_epoch=False, rank_zero_only=True)
        return loss * batch_size

    def configure_optimizers(self):
        optimizer = create_optimizer(self.model, self.cfg.task.optimizer)
        scheduler = create_scheduler(optimizer, self.cfg.task.scheduler)
        return [optimizer], [scheduler]
    
    # def train(self):
    #     optimizer = self.configure_optimizers()[0]
    #     epochs = 5
    #     print(self.train_dataloader)
    #     loss_fn = create_loss_function_cls(self.cfg)
    #     self.model.train()
    #     for epoch in range(epochs): 
    #         start = datetime.datetime.now()
    #         print("Start time of epoch", datetime.datetime.now())

    #         actual = []
    #         preds = []
    #         for index, batch in tqdm(enumerate(self.train_loader), desc="Files processed"):
    #             images, labels = batch
    #             for label in labels:
    #                 actual.append(np.argmax(label))
    #             predictions = self(images)["Main"]
    #             for prediction in predictions:
    #                 preds.append(np.argmax(prediction.detach().numpy()))
    #             loss = loss_fn(predictions, labels)
    #             batch_size = len(labels)
    #             loss = loss*batch_size
    #             loss.backward()
    #             optimizer.step()
    #         print(accuracy_score(actual, preds))
    #         end_time = datetime.datetime.now()
    #         print("Duration of epoch", end_time-start)
    #         print("End time of epoch", end_time)



class InferenceModel_CLS(BaseModel):
    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg
        # TODO: Add FastModel
        self.predict_loader = create_dataloader_cls(cfg.task.data, cfg.dataset, cfg.task.task)

    def predict_dataloader(self):
        return self.predict_loader

    def predict_step(self, batch, batch_idx):
        images, labels = batch
        scores = self(images)
        print(scores["Main"])
        predicts = torch.argmax(scores["Main"], dim=1)
        print(predicts)
        return predicts
        
        if getattr(self.predict_loader, "is_stream", None):
            fps = self._display_stream(img)
        else:
            fps = None
        if getattr(self.cfg.task, "save_predict", None):
            self._save_image(img, batch_idx)
        return img, fps