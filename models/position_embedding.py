# positional embedding
## height and weight ratio times 2pi separately,
## each used to calculate a tensor, then concatenate them together

import math
import torch
from torch import nn

class PositionEmbedding(nn.Module):
    def __init__(self, height, width, d_model, normalize=True):
        super().__init__()
        self.height = height
        self.width = width
        self.d_model = d_model
        self.normalize = normalize
        self.temperature=10000

        ## compute the positional embedding
        area=torch.ones((height, width), dtype=torch.float32)
        y_embed = area.cumsum(0, dtype=torch.float32)
        x_embed = area.cumsum(1, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[-1:, :] + eps) * 2 * math.pi
            x_embed = x_embed / (x_embed[:, -1:] + eps) * 2 * math.pi

        dim_t = torch.arange(self.d_model // 2, dtype=torch.float32)
        dim_t = self.temperature ** (2 * (dim_t // 2) / (self.d_model//2))

        pos_x = x_embed[:, :, None] / dim_t
        pos_y = y_embed[:, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
        pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
        self.pe = torch.cat((pos_y, pos_x), dim=2).permute(2, 0, 1).unsqueeze(0)  # [1, d_model, height, width]

    def forward(self, x):
        return self.pe.to(x.device)
