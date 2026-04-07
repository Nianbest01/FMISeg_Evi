from net.model import SegModel
from net.uncertainty import nig_loss, get_uncertainty
from monai.losses import DiceCELoss
from torchmetrics import Accuracy, Dice
from torchmetrics.classification import BinaryJaccardIndex
import torch
import torch.nn as nn
import pytorch_lightning as pl
from copy import deepcopy
import pandas as pd
import sys
import numpy as np
import datetime

class CreateModel(pl.LightningModule):

    def pretty_print(self, dic, stage):
        epoch = dic.get("epoch", "-")

        msg = (
            f"[{stage.upper():5}] "
            f"Epoch {epoch:<3} | "
            f"Loss: {dic.get(stage + '_loss', 0):.4f} | "
            f"Dice: {dic.get(stage + '_dice', 0):.4f} | "
            f"mIoU: {dic.get(stage + '_MIoU', 0):.4f} | "
            f"Alea: {dic.get(stage + '_aleatoric_mean', 0):.4f} | "
            f"Epis: {dic.get(stage + '_epistemic_mean', 0):.4f}"
        )

        self.print(msg)
        
    def __init__(self, args):
        super(CreateModel, self).__init__()
        self.model   = SegModel(args.bert_type, args.vision_type, args.project_dim)
        self.lr      = args.lr
        self.history = {}

        # ── 分割主损失 ────────────────────────────────────────────────────
        self.loss_fn = DiceCELoss()

        # ── NIG 不确定性损失超参数 ────────────────────────────────────────
        self.nig_lambda_reg   = getattr(args, 'nig_lambda_reg',   5e-2)
        self.nig_lambda_total = getattr(args, 'nig_lambda_total', 1e-2)

        metrics_dict = {
            "acc":  Accuracy(task='binary'),
            "dice": Dice(),
            "MIoU": BinaryJaccardIndex(),
        }
        self.train_metrics = nn.ModuleDict(metrics_dict)
        self.val_metrics   = deepcopy(self.train_metrics)
        self.test_metrics  = deepcopy(self.train_metrics)
        self.save_hyperparameters()

    # ── 优化器与学习率调度 ────────────────────────────────────────────────
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=200, eta_min=1e-6)
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    def forward(self, x):
        return self.model.forward(x)

    # ── 共享单步逻辑 ──────────────────────────────────────────────────────
    def shared_step(self, batch, batch_idx):
        x, y = batch
        # 前向传播，返回 preds, preds2, final_nig
        preds, preds2, final_nig = self(x)

        # ── 分割损失 ──────────────────────────────────────────────────────
        loss1    = self.loss_fn(preds,  y)
        loss2    = self.loss_fn(preds2, y)
        loss_seg = loss1 + loss2

        # ── NIG 回归目标：Dice 误差 ───────────────────────────────────────
        with torch.no_grad():
            intersection = (preds.detach() * y.float()).sum(dim=(1,2,3))
            dice_score   = (2 * intersection + 1e-6) / (
                preds.detach().sum(dim=(1,2,3)) + y.float().sum(dim=(1,2,3)) + 1e-6)
            y_target = (1.0 - dice_score).unsqueeze(1)      # [B, 1]

        # ── 最终融合 NIG 损失 ────────────────────────────────────────────
        loss_nig, loss_nig_nll, loss_nig_reg = nig_loss(
            y_target, final_nig,
            lam=self.nig_lambda_reg,
            reduction='mean',
        )

        # ── 总损失 ────────────────────────────────────────────────────────
        loss = loss_seg + self.nig_lambda_total * loss_nig
        # loss = loss_seg

        return {
            'loss':         loss,
            'loss_seg':     loss_seg.detach(),
            'loss_nig':     loss_nig.detach(),
            # 'loss_nig_nll': loss_nig_nll.detach(),
            # 'loss_nig_reg': loss_nig_reg.detach(),
            'preds':        preds.detach(),
            'y':            y.detach(),
            'final_nig':    tuple(p.detach() for p in final_nig),
        }

    # ── 训练 / 验证 / 测试步骤 ────────────────────────────────────────────
    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx)

    def predict_step(self, batch, batch_idx):
        if isinstance(batch, list) and len(batch) == 2:
            preds, preds2, final_nig = self(batch[0])
        else:
            preds, preds2, final_nig = self(batch)
        return preds, final_nig

    # ── 步骤结束：计算 metrics，并将 final_nig 透传给 epoch_end ──────────
    def shared_step_end(self, outputs, stage):
        metrics = (self.train_metrics if stage == "train"
                   else (self.val_metrics if stage == "val"
                         else self.test_metrics))

        for name in metrics:
            step_metric = metrics[name](outputs['preds'], outputs['y']).item()
            if stage == "train":
                self.log(name, step_metric, prog_bar=True)

        # 训练阶段实时记录不确定性分解损失
        if stage == "train":
            self.log("train_loss", outputs["loss"], on_step=False, on_epoch=True, prog_bar=True)
            self.log("train_loss_seg", outputs['loss_seg'], on_step=False, on_epoch=True, prog_bar=True)
            self.log("train_loss_nig", outputs['loss_nig'], on_step=False, on_epoch=True)
            # self.log("train_loss_nig_nll", outputs['loss_nig_nll'], on_step=False, on_epoch=True)
            # self.log("train_loss_nig_reg", outputs['loss_nig_reg'], on_step=False, on_epoch=True)

        loss_key = "loss" if stage == "train" else (
            "val_loss" if stage == "val" else "test_loss")

        return {
            loss_key:    outputs["loss"].mean(),
            'final_nig': outputs['final_nig'],
        }

    def training_step_end(self, outputs):
        return self.shared_step_end(outputs, "train")

    def validation_step_end(self, outputs):
        return self.shared_step_end(outputs, "val")

    def test_step_end(self, outputs):
        return self.shared_step_end(outputs, "test")

    # ── Epoch 结束：汇总 metrics + 不确定性统计 ───────────────────────────
    def shared_epoch_end(self, outputs, stage="train"):
        metrics = (self.train_metrics if stage == "train"
                   else (self.val_metrics if stage == "val"
                         else self.test_metrics))

        epoch      = self.trainer.current_epoch
        stage_loss = torch.mean(torch.tensor(
            [t[(stage + "_loss").replace('train_', '')] for t in outputs]
        )).item()

        dic = {"epoch": epoch, stage + "_loss": stage_loss}

        for name in metrics:
            epoch_metric = metrics[name].compute().item()
            metrics[name].reset()
            dic[stage + "_" + name] = epoch_metric

        # ── 不确定性统计 ──────────────────────────────────────────────────
        nig_u     = torch.cat([t['final_nig'][0] for t in outputs], dim=0)
        nig_la    = torch.cat([t['final_nig'][1] for t in outputs], dim=0)
        nig_alpha = torch.cat([t['final_nig'][2] for t in outputs], dim=0)
        nig_beta  = torch.cat([t['final_nig'][3] for t in outputs], dim=0)

        aleatoric, epistemic = get_uncertainty((nig_u, nig_la, nig_alpha, nig_beta))

        dic[stage + "_aleatoric_mean"] = aleatoric.mean().item()
        dic[stage + "_epistemic_mean"] = epistemic.mean().item()
        dic[stage + "_aleatoric_std"]  = aleatoric.std().item()
        dic[stage + "_epistemic_std"]  = epistemic.std().item()

        if stage != 'test':
            self.history[epoch] = dict(self.history.get(epoch, {}), **dic)
        return dic

    def training_epoch_end(self, outputs):
        dic = self.shared_epoch_end(outputs, stage="train")
        self.pretty_print(dic, "train")
        dic.pop("epoch", None)
        self.log_dict(dic, logger=True)

    def validation_epoch_end(self, outputs):
        dic = self.shared_epoch_end(outputs, stage="val")
        self.print_bar()
        self.pretty_print(dic, "val")

        dic.pop("epoch", None)
        self.log_dict(dic, logger=True)

        # 打印当前 best
        ckpt_cb = self.trainer.checkpoint_callback
        if ckpt_cb.best_model_score is not None:
            self.print(
                f"[BEST SO FAR] {ckpt_cb.monitor}: {ckpt_cb.best_model_score:.4f}"
            )

    def test_epoch_end(self, outputs):
        dic = self.shared_epoch_end(outputs, stage="test")
        self.pretty_print(dic, "test")
        dic.pop("epoch", None)
        self.log_dict(dic, logger=True)

    def get_history(self):
        return pd.DataFrame(self.history.values())

    def print_bar(self):
        nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.print("\n" + "=" * 80 + "%s" % nowtime)
