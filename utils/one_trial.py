from utils.LitAuto import LitAuto
from torch.utils.data import DataLoader, Dataset
import torch
import lightning.pytorch as pl
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger, MLFlowLogger
from typing import Literal
from typing import Optional


class LitModel(object):
    def __init__(self, estimator, hparams: dict, model_params: dict) -> None:
        """
        batchsize

        seed            default:1024

        clip_value

        clip_algo

        max_epochs      default:200
        """
        self.model_params = model_params
        self.hparams = hparams
        self.estimator = estimator
        self.update_params()

    def set_ckptdir(self, ckptdir):
        self.hparams["ckptdir"] = ckptdir
        self.__ckptdir = ckptdir

    def set_ckptname(self, ckptname):
        self.hparams["ckptname"] = ckptname
        self.__ckptname = ckptname

    def update_params(self, model_params: dict = {}, hparams: dict = {}):
        self.model_params.update(model_params)
        self.hparams.update(hparams)

        self.__name = self.hparams.get("name", None)
        self.__batchsize = self.hparams["batchsize"]
        self.__seed = self.hparams.get("seed", 1024)
        self.__max_epochs = self.hparams.get("max_epochs", 400)
        self.__clip_value = self.hparams.get("clip_value", None)
        self.__clip_algo = self.hparams.get("clip_algo", None)
        self.__monitor = self.hparams.get("monitor", "valid_mcc")
        self.__mode = self.hparams.get("mode", "max")
        self.__patience = self.hparams.get("patience", 30)
        self.__ckptdir = self.hparams.get("ckptdir", "ckpt")
        self.__ckptname = self.hparams.get("ckptname", self.__name)
        self.__enable_model_summary = self.hparams.get("model_summary", False)
        self.__enable_progress_bar = self.hparams.get("progress_bar", False)
        self.__logdir = self.hparams.get("logdir", None)
        self.__logname = self.hparams.get("logname", self.__name)
        self.__logtype = self.hparams.get("logtype", "TensorBoard")
        self.__model_type = self.hparams.get("model_type", LitAuto)

    def register_logger(self, type_logger: Literal["TensorBoard", "Wandb", "MLFlow"]):
        if type_logger == "TensorBoard":
            return TensorBoardLogger(
                save_dir=self.__logdir if self.__logdir else ".\\tensorboard",
                name=self.__logname if self.__logname else "lightning_logs",
            )
        elif type_logger == "Wandb":
            return WandbLogger(name=self.__logname, save_dir=self.__logdir if self.__logdir else ".\\wandb")
        elif type_logger == "MLFlow":
            return MLFlowLogger(
                experiment_name=self.__logname, save_dir=self.__logdir if self.__logdir else ".\\mlruns"
            )
        else:
            return None

    def refresh(self):
        earlys = EarlyStopping(self.__monitor, patience=self.__patience, mode=self.__mode)
        ckpt = ModelCheckpoint(self.__ckptdir, self.__ckptname, self.__monitor, mode=self.__mode)
        pl.seed_everything(self.__seed, workers=True)
        self.net = self.__model_type(self.estimator, self.model_params, **self.hparams)
        self.trainer = L.Trainer(
            gradient_clip_val=self.__clip_value,
            gradient_clip_algorithm=self.__clip_algo,
            logger=self.register_logger(self.__logtype),
            accelerator="gpu",
            max_epochs=self.__max_epochs,
            callbacks=[ckpt, earlys],
            enable_progress_bar=self.__enable_progress_bar,
            enable_model_summary=self.__enable_model_summary,
            # deterministic="warn",
        )

    def fit(self, train_data: Dataset, valid_data: Optional[Dataset] = None, verbose=False):
        self.train_set, self.valid_set = train_data, valid_data
        self.refresh()
        train_iter = DataLoader(
            self.train_set, self.__batchsize if self.__batchsize > 0 else len(self.train_set), shuffle=True  # type: ignore
        )
        if self.valid_set:
            valid_iter = DataLoader(
                self.valid_set, self.__batchsize if self.__batchsize > 0 else len(self.valid_set), shuffle=False  # type: ignore
            )
        else:
            valid_iter = None
        self.trainer.fit(self.net, train_dataloaders=train_iter, val_dataloaders=valid_iter)
        if valid_iter is not None:
            return self.trainer.test(self.net, dataloaders=valid_iter, ckpt_path="best", verbose=verbose)[0]
        else:
            return self.trainer.test(self.net, dataloaders=train_iter, ckpt_path="best", verbose=verbose)[0]

    def predict_proba(self, dataset: Dataset, is_loader=False, ckpt_path="best"):
        if is_loader:
            data_loader = dataset
        else:
            data_loader = DataLoader(dataset, self.__batchsize if self.__batchsize > 0 else len(dataset), shuffle=False)  # type: ignore
        if not hasattr(self, "trainer"):
            self.refresh()
        return torch.concat(self.trainer.predict(self.net, dataloaders=data_loader, ckpt_path=ckpt_path), dim=0)  # type: ignore

    def test(self, test_data: Dataset, is_loader=False, verbose=False, ckpt_path="best"):
        if is_loader:
            test_loader = test_data
        else:
            test_loader = DataLoader(
                test_data, self.__batchsize if self.__batchsize > 0 else len(test_data), shuffle=False  # type: ignore
            )
        if not hasattr(self, "trainer"):
            self.refresh()
        return self.trainer.test(self.net, dataloaders=test_loader, ckpt_path=ckpt_path, verbose=verbose)[0]

    @property
    def ckpt_path(self):
        return self.trainer.ckpt_path
