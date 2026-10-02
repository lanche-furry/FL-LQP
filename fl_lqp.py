"""FL-LQP: Frozen-LLM Learnable Query Pooling."""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional


@dataclass
class FL_LQPConfig:
    hidden_size: int = 2560
    num_heads: int = 8


class QueryAttentionPool(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int = 8):
        super().__init__()
        self.hidden_size = hidden_size
        self.q = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02)
        self.pos_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x, mask=None):
        B = x.shape[0]
        x_f = x.float()
        if mask is not None:
            mf = mask.unsqueeze(-1).float()
            avg = (x_f * mf).sum(1, keepdim=True) / mf.sum(1, keepdim=True).clamp(min=1.0)
        else:
            avg = x_f.mean(1, keepdim=True)
        q = (self.q.float() + self.pos_proj(avg)).expand(B, -1, -1).contiguous()
        kp = (~mask.bool()) if mask is not None else None
        out, _ = self.attn(q, x_f, x_f, key_padding_mask=kp)
        return self.norm(out[:, 0, :]).to(x.dtype)


class FL_LQPModel(nn.Module):
    def __init__(self, backbone, cfg=None):
        super().__init__()
        self.backbone = backbone
        hs = backbone.config.hidden_size
        self.cfg = cfg or FL_LQPConfig(hidden_size=hs)
        self.pool = QueryAttentionPool(self.cfg.hidden_size, self.cfg.num_heads)
        for p in self.backbone.parameters():
            p.requires_grad = False

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        return self.pool(out.last_hidden_state, attention_mask)

    def trainable_parameters(self):
        return (p for p in self.parameters() if p.requires_grad)
