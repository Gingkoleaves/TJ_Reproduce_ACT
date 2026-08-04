# transformer
## copy from preceding transformer_reproduce

from __future__ import annotations

import copy
import math
from typing import Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """Scaled dot-product multi-head attention, batch-first internally."""

    def __init__(self, d_model: int, h: int, dropout: float = 0.1):
        super().__init__()
        if d_model % h:
            raise ValueError(f'd_model ({d_model}) must be divisible by h ({h})')
        self.d_model = d_model
        self.h = h
        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: Tensor) -> Tensor:
        b, t, _ = x.shape
        return x.view(b, t, self.h, self.d_k).transpose(1, 2)

    def _merge_heads(self, x: Tensor) -> Tensor:
        b, _, t, _ = x.shape
        return x.transpose(1, 2).contiguous().view(b, t, self.d_model)

    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None):
        q = self._split_heads(self.w_q(query))
        k = self._split_heads(self.w_k(key))
        v = self._split_heads(self.w_v(value))
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask, float('-inf'))
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask[:, None, None, :], float('-inf'))
        attention = self.dropout(torch.softmax(scores, dim=-1))
        return self.w_o(self._merge_heads(torch.matmul(attention, v)))


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


def _add_pos(x: Tensor, pos: Optional[Tensor]) -> Tensor:
    return x if pos is None else x + pos


class TransformerEncoderLayer(nn.Module):
    """ACT-compatible sequence-first wrapper around the handmade blocks."""

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation='relu', normalize_before=False):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, dim_feedforward, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.normalize_before = normalize_before
        self.activation = activation

    def forward_post(self, src, src_key_padding_mask=None, pos=None):
        batch_src = src.transpose(0, 1)
        q = _add_pos(batch_src, None if pos is None else pos.transpose(0, 1))
        attn = self.self_attn(q, q, batch_src, key_padding_mask=src_key_padding_mask)
        batch_src = self.norm1(batch_src + self.dropout1(attn))
        batch_src = self.norm2(batch_src + self.dropout2(self.feed_forward(batch_src)))
        return batch_src.transpose(0, 1)

    def forward_pre(self, src, src_key_padding_mask=None, pos=None):
        batch_src = src.transpose(0, 1)
        norm_src = self.norm1(batch_src)
        p = None if pos is None else pos.transpose(0, 1)
        attn = self.self_attn(_add_pos(norm_src, p), _add_pos(norm_src, p), norm_src,
                              key_padding_mask=src_key_padding_mask)
        batch_src = batch_src + self.dropout1(attn)
        norm_src = self.norm2(batch_src)
        batch_src = batch_src + self.dropout2(self.feed_forward(norm_src))
        return batch_src.transpose(0, 1)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        del src_mask
        if self.normalize_before:
            return self.forward_pre(src, src_key_padding_mask, pos)
        return self.forward_post(src, src_key_padding_mask, pos)


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, mask=None, src_key_padding_mask=None, pos=None):
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask, pos=pos)
        return self.norm(output) if self.norm is not None else output


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation='relu', normalize_before=False):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.cross_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, dim_feedforward, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.normalize_before = normalize_before

    def forward_post(self, tgt, memory, tgt_key_padding_mask=None,
                     memory_key_padding_mask=None, pos=None, query_pos=None,
                     tgt_mask=None):
        batch_tgt = tgt.transpose(0, 1)
        batch_memory = memory.transpose(0, 1)
        qpos = None if query_pos is None else query_pos.transpose(0, 1)
        mpos = None if pos is None else pos.transpose(0, 1)
        q = _add_pos(batch_tgt, qpos)
        self_attn = self.self_attn(q, q, batch_tgt,
                                   key_padding_mask=tgt_key_padding_mask,
                                   attn_mask=tgt_mask)
        batch_tgt = self.norm1(batch_tgt + self.dropout1(self_attn))
        cross = self.cross_attn(_add_pos(batch_tgt, qpos), _add_pos(batch_memory, mpos),
                                batch_memory, key_padding_mask=memory_key_padding_mask)
        batch_tgt = self.norm2(batch_tgt + self.dropout2(cross))
        batch_tgt = self.norm3(batch_tgt + self.dropout3(self.feed_forward(batch_tgt)))
        return batch_tgt.transpose(0, 1)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None,
                pos=None, query_pos=None):
        del memory_mask
        if self.normalize_before:
            raise NotImplementedError('normalize_before is not supported by the handmade ACT blocks')
        return self.forward_post(tgt, memory, tgt_key_padding_mask,
                                 memory_key_padding_mask, pos, query_pos, tgt_mask)


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None,
                pos=None, query_pos=None):
        output = tgt
        intermediate = []
        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output) if self.norm is not None else output)
        if self.norm is not None:
            output = self.norm(output)
        if self.return_intermediate:
            return torch.stack(intermediate)
        return output.unsqueeze(0)


class Transformer(nn.Module):
    """ACT's image/state memory encoder and action-query decoder."""

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation='relu', normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers,
                                           nn.LayerNorm(d_model) if normalize_before else None)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers,
                                           nn.LayerNorm(d_model), return_intermediate_dec)
        self.d_model = d_model
        self.nhead = nhead
        self._reset_parameters()

    def _reset_parameters(self):
        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)

    @staticmethod
    def _repeat_position(pos, batch_size):
        if pos.size(1) == 1:
            return pos.repeat(1, batch_size, 1)
        if pos.size(1) != batch_size:
            raise ValueError(f'position batch {pos.size(1)} does not match input batch {batch_size}')
        return pos

    def forward(self, src, mask, query_embed, pos_embed, latent_input=None,
                proprio_input=None, additional_pos_embed=None):
        if src.dim() == 4:
            batch_size, _, _, _ = src.shape
            src = src.flatten(2).permute(2, 0, 1)
            pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
            pos_embed = self._repeat_position(pos_embed, batch_size)
            query_embed = query_embed.unsqueeze(1).repeat(1, batch_size, 1)
            if latent_input is None or proprio_input is None or additional_pos_embed is None:
                raise ValueError('image mode requires latent_input, proprio_input and additional_pos_embed')
            additional_pos = additional_pos_embed.unsqueeze(1).repeat(1, batch_size, 1)
            src = torch.cat((torch.stack((latent_input, proprio_input), dim=0), src), dim=0)
            pos_embed = torch.cat((additional_pos, pos_embed), dim=0)
        else:
            if src.dim() != 3:
                raise ValueError(f'expected src [B,S,D] or [B,C,H,W], got {tuple(src.shape)}')
            batch_size = src.size(0)
            src = src.permute(1, 0, 2)
            pos_embed = pos_embed.unsqueeze(1) if pos_embed.dim() == 2 else pos_embed
            pos_embed = self._repeat_position(pos_embed, batch_size)
            query_embed = query_embed.unsqueeze(1).repeat(1, batch_size, 1)

        tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        hidden = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                              pos=pos_embed, query_pos=query_embed)
        return hidden.transpose(1, 2)


def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )
