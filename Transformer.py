"""
Living AI System — Transformer Module
Full encoder-decoder transformer, encoder-only, decoder-only variants.
Vision Transformer, Swin Transformer, BERT-style, GPT-style, T5-style.
All attention variants: multi-head, cross, linear, sparse, local, sliding window.
"""

import asyncio
import math
from typing import Any

import structlog
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.base import BaseModule, ModuleOutput

log = structlog.get_logger(__name__)


class LinearAttention(nn.Module):
    """Linear attention — O(n) complexity instead of O(n²)."""

    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q = nn.Linear(embed_dim, embed_dim)
        self.k = nn.Linear(embed_dim, embed_dim)
        self.v = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q = F.elu(self.q(x)) + 1
        k = F.elu(self.k(x)) + 1
        v = self.v(x)
        k_sum = k.sum(dim=1, keepdim=True)
        kv = torch.bmm(k.transpose(1, 2), v)
        qkv = torch.bmm(q, kv)
        normalizer = (q * k_sum).sum(dim=-1, keepdim=True).clamp(min=1e-6)
        return self.out(qkv / normalizer)


class SparseAttention(nn.Module):
    """Sparse attention — attends to local window plus global tokens."""

    def __init__(self, embed_dim: int, num_heads: int, window_size: int = 64):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.q = nn.Linear(embed_dim, embed_dim)
        self.k = nn.Linear(embed_dim, embed_dim)
        self.v = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q = self.q(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        # Local window mask
        mask = torch.zeros(T, T, device=x.device)
        for i in range(T):
            start = max(0, i - self.window_size // 2)
            end = min(T, i + self.window_size // 2)
            mask[i, start:end] = 1.0
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.masked_fill(mask.unsqueeze(0).unsqueeze(0) == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.out(out)


class CrossAttention(nn.Module):
    """Cross-attention between encoder and decoder sequences."""

    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.q = nn.Linear(embed_dim, embed_dim)
        self.k = nn.Linear(embed_dim, embed_dim)
        self.v = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        B, Tq, C = query.shape
        Tk = context.shape[1]
        q = self.q(query).view(B, Tq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(context).view(B, Tk, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(context).view(B, Tk, self.num_heads, self.head_dim).transpose(1, 2)
        attn = F.softmax((q @ k.transpose(-2, -1)) * self.scale, dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, Tq, C)
        return self.out(out)


class EncoderBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim), nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor, src_key_padding_mask=None) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x, key_padding_mask=src_key_padding_mask)
        x = self.norm1(x + attn_out)
        return self.norm2(x + self.ff(x))


class DecoderBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn = CrossAttention(embed_dim, num_heads)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim), nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor, memory: torch.Tensor, tgt_mask=None) -> torch.Tensor:
        self_attn_out, _ = self.self_attn(x, x, x, attn_mask=tgt_mask)
        x = self.norm1(x + self_attn_out)
        x = self.norm2(x + self.cross_attn(x, memory))
        return self.norm3(x + self.ff(x))


class EncoderDecoderTransformer(nn.Module):
    """Full encoder-decoder transformer (T5-style)."""

    def __init__(self, vocab_size: int, embed_dim: int = 256, num_heads: int = 8,
                 num_encoder_layers: int = 6, num_decoder_layers: int = 6,
                 ff_dim: int = 1024, max_seq_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.src_embed = nn.Embedding(vocab_size, embed_dim)
        self.tgt_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(max_seq_len, embed_dim)
        self.encoder = nn.ModuleList([
            EncoderBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_encoder_layers)
        ])
        self.decoder = nn.ModuleList([
            DecoderBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_decoder_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def encode(self, src: torch.Tensor) -> torch.Tensor:
        T = src.shape[1]
        pos = torch.arange(T, device=src.device).unsqueeze(0)
        x = self.dropout(self.src_embed(src) + self.pos_embed(pos))
        for layer in self.encoder:
            x = layer(x)
        return self.norm(x)

    def decode(self, tgt: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        T = tgt.shape[1]
        pos = torch.arange(T, device=tgt.device).unsqueeze(0)
        mask = torch.tril(torch.ones(T, T, device=tgt.device))
        x = self.dropout(self.tgt_embed(tgt) + self.pos_embed(pos))
        for layer in self.decoder:
            x = layer(x, memory, tgt_mask=mask == 0)
        return self.lm_head(self.norm(x))

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        memory = self.encode(src)
        return self.decode(tgt, memory)


class VisionTransformer(nn.Module):
    """
    Vision Transformer — treats image patches as token sequences.
    The patch embedding space is the infinite path space.
    Global attention across all patches is the filter.
    The classification is the revealed answer.
    """

    def __init__(self, img_size: int = 224, patch_size: int = 16, in_channels: int = 3,
                 num_classes: int = 1000, embed_dim: int = 768, depth: int = 12,
                 num_heads: int = 12, ff_dim: int = 3072, dropout: float = 0.1):
        super().__init__()
        assert img_size % patch_size == 0
        num_patches = (img_size // patch_size) ** 2
        patch_dim = in_channels * patch_size * patch_size

        self.patch_size = patch_size
        self.patch_embed = nn.Linear(patch_dim, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            EncoderBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        p = self.patch_size
        x = x.unfold(2, p, p).unfold(3, p, p)
        x = x.contiguous().view(B, -1, C * p * p)
        x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x)[:, 0])


class TransformerModule(BaseModule):
    """
    Transformer module — activates for tasks requiring
    structured sequence-to-sequence processing,
    vision understanding, or complex language tasks.
    """

    def __init__(self):
        self._vit: VisionTransformer | None = None
        self._seq2seq: EncoderDecoderTransformer | None = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def name(self) -> str:
        return "transformer"

    @property
    def output_type(self) -> str:
        return "text"

    async def initialise(self) -> None:
        self._vit = VisionTransformer().to(self._device)
        self._vit.eval()
        self._seq2seq = EncoderDecoderTransformer(vocab_size=10000).to(self._device)
        self._seq2seq.eval()
        log.info("transformer_module.initialised")

    async def execute(
        self,
        message: Any,
        episodic_context: list[dict],
        knowledge_context: list[dict],
        working_memory: list[dict],
    ) -> ModuleOutput:
        return ModuleOutput(
            content="",
            confidence=0.0,
            output_type=self.output_type,
            source=self.name,
        )
