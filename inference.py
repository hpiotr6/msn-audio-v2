from pathlib import Path
import torch


import torchvision
from lightly.models.modules.masked_autoencoder import MAEBackbone
from torch import nn

vit = torchvision.models.vit_b_32(pretrained=False)
vit.conv_proj = nn.Sequential(nn.Conv2d(1, 3, kernel_size=1), vit.conv_proj)
backbone = MAEBackbone.from_vit(vit)
device = "gpu" if torch.cuda.is_available() else "cpu"
state_dict = torch.load("model-epoch=470-train_loss=5.49.ckpt", map_location=device)[
    "state_dict"
]
filtered_state_dict = {
    key.replace("backbone.", ""): value
    for key, value in state_dict.items()
    if key.startswith("backbone")
}
backbone.load_state_dict(filtered_state_dict)
