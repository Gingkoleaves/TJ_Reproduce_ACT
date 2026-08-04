# Top component of ACT
## Combines visual Transformer, DETR-style action queries, and CVAE latent sampling.

from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from .position_embedding import PositionEmbedding
from .transformer import TransformerEncoder, TransformerEncoderLayer, build_transformer
from .vision_encoder import VisionEncoder


def reparametrize(mu: Tensor, logvar: Tensor) -> Tensor:
    """Sample z with the reparameterization trick."""
    std = torch.exp(0.5 * logvar)
    return mu + torch.randn_like(std) * std


def get_sinusoid_encoding_table(n_position: int, d_hid: int) -> Tensor:
    """Sinusoidal table used by the posterior action encoder."""
    position = torch.arange(n_position, dtype=torch.float32).unsqueeze(1)
    hid = torch.arange(d_hid, dtype=torch.float32).unsqueeze(0)
    angle = position / torch.pow(torch.tensor(10000.0), 2 * torch.div(hid, 2, rounding_mode='floor') / d_hid)
    table = torch.zeros(n_position, d_hid)
    table[:, 0::2] = torch.sin(angle[:, 0::2])
    table[:, 1::2] = torch.cos(angle[:, 1::2])
    return table.unsqueeze(0)


class DETRVAE(nn.Module):
    """ACT CVAE + visual Transformer action predictor."""

    def __init__(self, backbones, transformer, encoder, state_dim=14,
                 num_queries=100, camera_names: Sequence[str] = ('top',),
                 latent_dim=32):
        super().__init__()
        self.num_queries = num_queries
        self.camera_names = tuple(camera_names)
        self.transformer = transformer
        self.encoder = encoder
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        hidden_dim = transformer.d_model

        self.action_head = nn.Linear(hidden_dim, state_dim)
        self.is_pad_head = nn.Linear(hidden_dim, 1)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        if backbones is not None:
            if len(backbones) == 0:
                raise ValueError('backbones must contain at least one visual encoder')
            self.input_proj = nn.Conv2d(backbones[0].num_channels, hidden_dim, kernel_size=1)
            self.backbones = nn.ModuleList(backbones)
            self.input_proj_robot_state = nn.Linear(state_dim, hidden_dim)
        else:
            self.backbones = None
            self.input_proj_robot_state = nn.Linear(state_dim, hidden_dim)
            self.input_proj_env_state = nn.Linear(7, hidden_dim)
            self.pos = nn.Embedding(2, hidden_dim)

        # Posterior encoder: [CLS, qpos, action_1 ... action_K] -> z.
        self.cls_embed = nn.Embedding(1, hidden_dim)
        self.encoder_action_proj = nn.Linear(state_dim, hidden_dim)
        self.encoder_joint_proj = nn.Linear(state_dim, hidden_dim)
        self.latent_proj = nn.Linear(hidden_dim, latent_dim * 2)
        self.register_buffer(
            'pos_table', get_sinusoid_encoding_table(2 + num_queries, hidden_dim)
        )

        # Decoder input tokens: latent z and proprioception.
        self.latent_out_proj = nn.Linear(latent_dim, hidden_dim)
        self.additional_pos_embed = nn.Embedding(2, hidden_dim)

    def _encode_posterior(self, qpos: Tensor, actions: Tensor, is_pad: Tensor):
        batch_size = qpos.size(0)
        actions = actions[:, :self.num_queries]
        is_pad = is_pad[:, :self.num_queries].bool()

        action_embed = self.encoder_action_proj(actions)
        qpos_embed = self.encoder_joint_proj(qpos).unsqueeze(1)
        cls_embed = self.cls_embed.weight.unsqueeze(0).expand(batch_size, -1, -1)
        encoder_input = torch.cat((cls_embed, qpos_embed, action_embed), dim=1).transpose(0, 1)

        prefix_mask = torch.zeros(batch_size, 2, dtype=torch.bool, device=qpos.device)
        encoder_mask = torch.cat((prefix_mask, is_pad), dim=1)
        seq_len = encoder_input.size(0)
        pos = self.pos_table[:, :seq_len].detach().to(qpos.device).transpose(0, 1)
        encoder_output = self.encoder(
            encoder_input, pos=pos, src_key_padding_mask=encoder_mask
        )[0]
        latent_info = self.latent_proj(encoder_output)
        mu, logvar = latent_info[:, :self.latent_dim], latent_info[:, self.latent_dim:]
        return mu, logvar

    def _visual_memory(self, qpos: Tensor, image: Tensor):
        if image.dim() != 5:
            raise ValueError(f'image must be [B,Ncam,3,H,W], got {tuple(image.shape)}')
        if image.size(1) != len(self.camera_names):
            raise ValueError(
                f'image has {image.size(1)} cameras, expected {len(self.camera_names)}'
            )

        all_features, all_positions = [], []
        for camera_id in range(image.size(1)):
            features, positions = self.backbones[0](image[:, camera_id])
            all_features.append(self.input_proj(features[0]))
            all_positions.append(positions[0])

        # Match ACT: multiple camera feature maps are concatenated along width.
        src = torch.cat(all_features, dim=3)
        pos = torch.cat(all_positions, dim=3)
        proprio = self.input_proj_robot_state(qpos)
        return src, pos, proprio

    def forward(self, qpos: Tensor, image: Optional[Tensor], env_state: Optional[Tensor],
                actions: Optional[Tensor] = None, is_pad: Optional[Tensor] = None):
        if qpos.dim() != 2 or qpos.size(-1) != self.state_dim:
            raise ValueError(f'qpos must be [B,{self.state_dim}], got {tuple(qpos.shape)}')
        batch_size = qpos.size(0)

        if actions is not None:
            if is_pad is None:
                raise ValueError('is_pad is required when actions are provided')
            mu, logvar = self._encode_posterior(qpos, actions, is_pad)
            latent_sample = reparametrize(mu, logvar)
        else:
            mu = logvar = None
            latent_sample = torch.zeros(
                batch_size, self.latent_dim, device=qpos.device, dtype=qpos.dtype
            )
        latent_input = self.latent_out_proj(latent_sample)

        if self.backbones is not None:
            src, pos, proprio = self._visual_memory(qpos, image)
            hidden = self.transformer(
                src, None, self.query_embed.weight, pos,
                latent_input, proprio, self.additional_pos_embed.weight
            )[0]
        else:
            if env_state is None:
                raise ValueError('env_state is required for the state-only model')
            state_tokens = torch.cat((self.input_proj_robot_state(qpos),
                                      self.input_proj_env_state(env_state)), dim=1).unsqueeze(1)
            hidden = self.transformer(
                state_tokens, None, self.query_embed.weight, self.pos.weight
            )[0]

        return self.action_head(hidden), self.is_pad_head(hidden), [mu, logvar]


def build_encoder(args):
    layer = TransformerEncoderLayer(
        d_model=args.hidden_dim,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        activation='relu',
        normalize_before=args.pre_norm,
    )
    norm = nn.LayerNorm(args.hidden_dim) if args.pre_norm else None
    return TransformerEncoder(layer, args.enc_layers, norm)


def build_vae(args):
    """Build ACT from the argument namespace used by the training script."""
    state_dim = getattr(args, 'state_dim', 14)
    camera_names = tuple(args.camera_names)
    hidden_dim = args.hidden_dim
    feature_height, feature_width = getattr(args, 'feature_size', (15, 20))
    position_encoder = PositionEmbedding(feature_height, feature_width, hidden_dim, normalize=True)
    backbone = VisionEncoder(
        getattr(args, 'backbone', 'resnet18'),
        position_encoder,
        pretrained=getattr(args, 'pretrained_backbone', True),
        dilation=getattr(args, 'dilation', False),
    )
    transformer = build_transformer(args)
    encoder = build_encoder(args)
    return DETRVAE(
        [backbone], transformer, encoder,
        state_dim=state_dim,
        num_queries=args.num_queries,
        camera_names=camera_names,
        latent_dim=getattr(args, 'latent_dim', 32),
    )


# Names used by ACT-style callers.
build = build_vae
build_ACT_model = build_vae
