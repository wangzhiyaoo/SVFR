
import torch
from diffusers import ModelMixin
from einops import rearrange
from torch import nn

class IDProjConvModel(ModelMixin):
    def __init__(self, in_channels=2048, out_channels=1024):
        super().__init__()

        self.project1024 = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False)
        self.final_norm = torch.nn.LayerNorm(out_channels)

    def forward(self, src_id_features_7_7_1024):
        c = self.project1024(src_id_features_7_7_1024)
        c = torch.flatten(c, 2)
        c = torch.transpose(c, 2, 1)
        c = self.final_norm(c)

        return c
