from evar.ar_base import (
    BaseAudioRepr,
    ToLogMelSpec,
    calculate_norm_stats,
    normalize_spectrogram,
    temporal_pooling,
)
import torch


import torchvision
from lightly.models.modules.masked_autoencoder import MAEBackbone
from torch import nn


class AR_MSN(BaseAudioRepr):
    def __init__(self, cfg):
        super().__init__(cfg=cfg)
        self.to_feature = ToLogMelSpec(cfg)
        vit = torchvision.models.vit_b_32(pretrained=False)
        vit.conv_proj = nn.Sequential(nn.Conv2d(1, 3, kernel_size=1), vit.conv_proj)
        self.body = MAEBackbone.from_vit(vit)

        # self.body = AudioNTT2022Encoder(n_mels=cfg.n_mels, d=cfg.feature_d)
        if cfg.weight_file is not None and cfg.weight_file != "":
            device = "cuda" if torch.cuda.is_available() else "cpu"
            state_dict = torch.load(cfg.weight_file, map_location=device)["state_dict"]

            filtered_state_dict = {
                key.replace("backbone.", ""): value
                for key, value in state_dict.items()
                if key.startswith("backbone")
            }
            self.body.load_state_dict(filtered_state_dict)

    def precompute(self, device, data_loader):
        self.norm_stats = calculate_norm_stats(device, data_loader, self.to_feature)

    def encode_frames(self, batch_audio):
        x = self.to_feature(batch_audio)
        x = normalize_spectrogram(self.norm_stats, x)  # B,F,T

        x = self.augment_if_training(x)
        x = x.unsqueeze(1)  # -> B,1,F,T
        x = self.pad(x)
        x = self.body(x)  # -> B,T,D=C*F
        # x = x.transpose(1, 2)  # -> B,D,T
        return x

    def pad(self, x):
        if x.size(-1) <= 576:
            desired_size = (64, 576)
        elif x.size(-1) <= 3136:
            desired_size = (64, 3136)
        assert x.size(-2) <= desired_size[0]

        # Calculate the amount of padding needed for each dimension
        pad_dim2 = max(0, desired_size[0] - x.size(-2))
        pad_dim3 = max(0, desired_size[1] - x.size(-1))

        # Pad the tensor
        x = torch.nn.functional.pad(x, (0, pad_dim3, 0, pad_dim2))
        return x

    def forward(self, batch_audio):
        x = self.encode_frames(batch_audio)
        # x = temporal_pooling(self, x)
        return x
