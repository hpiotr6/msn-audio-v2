from datetime import datetime
from pathlib import Path
from transforms import MultiViewTransform
import hydra

from omegaconf import DictConfig, OmegaConf

from byol_a2.dataset import WaveInLMSOutDataset
import copy

import pytorch_lightning as pl
import torch
import torchvision
from lightly.loss import MSNLoss
from lightly.models import utils
from lightly.models.modules.heads import MSNProjectionHead
from lightly.models.modules.masked_autoencoder import MAEBackbone
from torch import nn
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torch.optim import lr_scheduler


class MSN(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()

        self.mask_ratio = 0.15

        vit = torchvision.models.vit_b_32(pretrained=True)
        vit.conv_proj = nn.Sequential(nn.Conv2d(1, 3, kernel_size=1), vit.conv_proj)
        self.backbone = MAEBackbone.from_vit(vit)

        self.projection_head = MSNProjectionHead(768)

        self.anchor_backbone = copy.deepcopy(self.backbone)
        self.anchor_projection_head = copy.deepcopy(self.projection_head)

        utils.deactivate_requires_grad(self.backbone)
        utils.deactivate_requires_grad(self.projection_head)

        self.prototypes = nn.Linear(256, 1024, bias=False).weight
        self.criterion = MSNLoss()

        self.lr = cfg.lr
        self.milestones = cfg.scheduler.milestones
        self.gamma = cfg.scheduler.gamma

    def training_step(self, batch, batch_idx):
        utils.update_momentum(self.anchor_backbone, self.backbone, 0.996)
        utils.update_momentum(self.anchor_projection_head, self.projection_head, 0.996)

        # views = [one_channel_adapter(example) for example in batch]
        # views = [example for example in batch]
        views = [view.to(self.device, non_blocking=True) for view in batch]
        targets = views[0]
        anchors = views[1]
        anchors_focal = torch.concat(views[2:], dim=0)

        targets_out = self.backbone(targets)
        targets_out = self.projection_head(targets_out)
        anchors_out = self.encode_masked(anchors)
        anchors_focal_out = self.encode_masked(anchors_focal)
        anchors_out = torch.cat([anchors_out, anchors_focal_out], dim=0)

        loss = self.criterion(anchors_out, targets_out, self.prototypes.data)
        self.log("train_loss", loss, on_epoch=True)
        return loss

    def encode_masked(self, anchors):
        batch_size, _, _, width = anchors.shape
        seq_length = (width // self.anchor_backbone.patch_size) ** 2
        idx_keep, _ = utils.random_token_mask(
            size=(batch_size, seq_length),
            mask_ratio=self.mask_ratio,
            device=self.device,
        )
        out = self.anchor_backbone(anchors, idx_keep)
        return self.anchor_projection_head(out)

    def configure_optimizers(self):
        params = [
            *list(self.anchor_backbone.parameters()),
            *list(self.anchor_projection_head.parameters()),
            self.prototypes,
        ]
        optim = torch.optim.AdamW(params, lr=self.lr)

        # Create a learning rate scheduler with milestones
        scheduler = {
            "scheduler": lr_scheduler.MultiStepLR(
                optim, milestones=self.milestones, gamma=self.gamma
            ),
        }

        return [optim], [scheduler]


@hydra.main(version_base=None, config_path=".", config_name="config_v2.yaml")
def main(_cfg: DictConfig):
    cfg = OmegaConf.to_container(_cfg, resolve=True)

    # Add new keys or modify existing ones
    cfg["unit_samples"] = int(_cfg.sample_rate * _cfg.unit_sec)
    cfg = OmegaConf.create(cfg)
    model = MSN(cfg)
    if cfg.wandb:
        today_date = datetime.today().strftime(r"%d-%m-%Y")
        project_name = f"audio-{today_date}"
        wandb_logger = WandbLogger(project=project_name, entity=cfg.entity)
        wandb_logger.experiment.config.update({**cfg})
        wandb_logger.watch(model)

    checkpoint_callback = ModelCheckpoint(
        monitor="train_loss",
        # dirpath="checkpoints/",
        filename="model-{epoch:02d}-{train_loss:.2f}",
        save_top_k=1,
        mode="min",
        save_last=True,
    )

    files = sorted(Path(cfg.audio_dir).glob("*.wav"))

    epoch_samples = len(files) // cfg.bs
    tr = MultiViewTransform(epoch_samples=epoch_samples)
    # ds = WaveInLMSOutDataset(cfg, files, labels=None, tfms=None)
    ds = WaveInLMSOutDataset(cfg, files, labels=None, tfms=tr, random_crop=True)

    dataloader = torch.utils.data.DataLoader(
        ds,
        batch_size=cfg.bs,
        shuffle=True,
        drop_last=True,
        num_workers=cfg.num_workers,
    )

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    trainer = pl.Trainer(
        max_epochs=cfg.epochs,
        devices=1,
        accelerator=accelerator,
        callbacks=[checkpoint_callback, lr_monitor],
        logger=wandb_logger if cfg.wandb else None,
        enable_progress_bar=False,
    )
    trainer.fit(model=model, train_dataloaders=dataloader, ckpt_path=cfg.resume)


if __name__ == "__main__":
    main()
