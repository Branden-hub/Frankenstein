"""
Living AI System — Language Module
The primary language understanding and generation module.
This is the core text processing engine — it is always activated
for text inputs and provides the primary response.
Built on the transformer architecture as specified in the framework.
Trained from scratch on this system — no external model dependencies.
"""

import asyncio
import json
import math
import time
from pathlib import Path
from typing import AsyncIterator, Any

import structlog
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.base import BaseModule, ModuleOutput

log = structlog.get_logger(__name__)

MODEL_PATH = Path("models/language_model.pt")
VOCAB_PATH = Path("models/vocabulary.json")
MAX_NEW_TOKENS = 2048
TEMPERATURE = 0.7
TOP_P = 0.9


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention mechanism.
    The attention pattern is the Game of Infinite Paths in action:
    all token relationships exist simultaneously in the attention matrix,
    the softmax is the filter, and the attended representation
    is the revealed answer.
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, T, C = x.shape

        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention — the filter operation
        attn = (q @ k.transpose(-2, -1)) * self.scale

        if mask is not None:
            attn = attn.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


class FeedForward(nn.Module):
    """Position-wise feed-forward network."""

    def __init__(self, embed_dim: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """Single transformer block with pre-norm architecture."""

    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.ff = FeedForward(embed_dim, ff_dim, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.ff(self.norm2(x))
        return x


class LanguageModelCore(nn.Module):
    """
    The core language model.
    A decoder-only transformer trained from scratch on this system.
    The token embedding space is the infinite path space.
    Each transformer block is a filter layer.
    The final output distribution is the revealed answer.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        max_seq_len: int,
        ff_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

        # Weight tying between embedding and lm_head
        self.lm_head.weight = self.token_embedding.weight

        # Initialise weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Xavier-style weight initialisation for stable training."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, T = input_ids.shape
        assert T <= self.max_seq_len, f"Sequence length {T} exceeds maximum {self.max_seq_len}"

        positions = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.dropout(
            self.token_embedding(input_ids) + self.position_embedding(positions)
        )

        # Apply causal mask if not provided
        if mask is None:
            mask = torch.tril(torch.ones(T, T, device=input_ids.device)).unsqueeze(0).unsqueeze(0)

        for block in self.blocks:
            x = block(x, mask)

        x = self.norm(x)
        logits = self.lm_head(x)
        return logits

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = MAX_NEW_TOKENS,
        temperature: float = TEMPERATURE,
        top_p: float = TOP_P,
    ) -> torch.Tensor:
        """
        Autoregressive generation with top-p sampling.
        The generation process is the Game of Infinite Paths:
        at each step all next-token paths exist simultaneously,
        temperature and top-p are the filters,
        and the selected token is revealed.
        """
        generated = input_ids.clone()

        for _ in range(max_new_tokens):
            # Truncate to max_seq_len if needed
            context = generated[:, -self.max_seq_len:]
            logits = self.forward(context)
            next_token_logits = logits[:, -1, :] / temperature

            # Top-p (nucleus) sampling
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs - F.softmax(sorted_logits, dim=-1) > top_p
            sorted_logits[sorted_indices_to_remove] = float("-inf")

            next_token_logits_filtered = next_token_logits.scatter(
                1, sorted_indices, sorted_logits
            )
            probs = F.softmax(next_token_logits_filtered, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)

            # Stop at end-of-sequence token (id 2)
            if next_token.item() == 2:
                break

        return generated


class SimpleTokenizer:
    """
    Character-level tokenizer as a starting point.
    The system learns better tokenization through training.
    """

    def __init__(self, vocab: dict[str, int] | None = None):
        if vocab:
            self.vocab = vocab
        else:
            # Build basic vocabulary
            chars = (
                list(" \n\t") +
                list("abcdefghijklmnopqrstuvwxyz") +
                list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") +
                list("0123456789") +
                list(".,!?;:'\"-()[]{}/@#$%^&*+=<>|\\`~_")
            )
            self.vocab = {"<pad>": 0, "<unk>": 1, "<eos>": 2, "<bos>": 3}
            for i, ch in enumerate(chars):
                self.vocab[ch] = i + 4

        self.reverse_vocab = {v: k for k, v in self.vocab.items()}

    def encode(self, text: str) -> list[int]:
        return [self.vocab.get(ch, self.vocab["<unk>"]) for ch in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.reverse_vocab.get(i, "") for i in ids if i not in {0, 2, 3})

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.vocab, f)

    @classmethod
    def load(cls, path: Path) -> "SimpleTokenizer":
        with open(path) as f:
            vocab = json.load(f)
        return cls(vocab=vocab)


class LanguageModule(BaseModule):
    """
    The primary language module.
    Always activated for text inputs.
    Provides the base response that other modules augment.
    """

    def __init__(self):
        self._model: LanguageModelCore | None = None
        self._tokenizer: SimpleTokenizer | None = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def name(self) -> str:
        return "language"

    @property
    def output_type(self) -> str:
        return "text"

    async def initialise(self) -> None:
        """Load or initialise the language model."""
        await asyncio.to_thread(self._load_or_init_model)
        log.info(
            "language_module.initialised",
            device=str(self._device),
            vocab_size=self._tokenizer.vocab_size if self._tokenizer else 0,
        )

    def _load_or_init_model(self) -> None:
        """Load model from disk or initialise a new one."""
        if VOCAB_PATH.exists():
            self._tokenizer = SimpleTokenizer.load(VOCAB_PATH)
        else:
            self._tokenizer = SimpleTokenizer()
            self._tokenizer.save(VOCAB_PATH)

        # Model configuration
        config = {
            "vocab_size": self._tokenizer.vocab_size,
            "embed_dim": 512,
            "num_heads": 8,
            "num_layers": 8,
            "max_seq_len": 2048,
            "ff_dim": 2048,
            "dropout": 0.1,
        }

        self._model = LanguageModelCore(**config)

        if MODEL_PATH.exists():
            try:
                state = torch.load(MODEL_PATH, map_location=self._device, weights_only=True)
                self._model.load_state_dict(state)
                log.info("language_module.model_loaded", path=str(MODEL_PATH))
            except Exception as exc:
                log.warning(
                    "language_module.model_load_failed",
                    error=str(exc),
                    note="Starting with fresh weights",
                )

        self._model.to(self._device)
        self._model.eval()

        total_params = sum(p.numel() for p in self._model.parameters())
        log.info("language_module.model_ready", parameters=total_params)

    def _build_prompt(
        self,
        content: str,
        working_memory: list[dict],
        episodic_context: list[dict],
        knowledge_context: list[dict],
    ) -> str:
        """
        Build the full prompt from all available context.
        All context is part of the path space that the model filters.
        """
        parts = []

        if knowledge_context:
            parts.append("KNOWLEDGE:")
            for k in knowledge_context[:3]:
                parts.append(f"- {k.get('content', '')[:200]}")

        if episodic_context:
            parts.append("MEMORY:")
            for e in episodic_context[:2]:
                parts.append(f"- {e.get('content', '')[:200]}")

        if working_memory:
            parts.append("CONVERSATION:")
            for msg in working_memory[-6:]:
                role = msg.get("role", "user").upper()
                parts.append(f"{role}: {msg.get('content', '')[:500]}")

        parts.append(f"USER: {content}")
        parts.append("ASSISTANT:")

        return "\n".join(parts)

    async def execute(
        self,
        message: Any,
        episodic_context: list[dict],
        knowledge_context: list[dict],
        working_memory: list[dict],
    ) -> ModuleOutput:
        """Generate a response to the input message."""
        if self._model is None or self._tokenizer is None:
            return ModuleOutput(
                content="Language model not initialised.",
                confidence=0.0,
                output_type=self.output_type,
                source=self.name,
                is_primary=True,
            )

        prompt = self._build_prompt(
            content=message.content,
            working_memory=working_memory,
            episodic_context=episodic_context,
            knowledge_context=knowledge_context,
        )

        response = await asyncio.to_thread(self._generate_response, prompt)

        return ModuleOutput(
            content=response,
            confidence=0.85,
            output_type=self.output_type,
            source=self.name,
            is_primary=True,
        )

    def _generate_response(self, prompt: str) -> str:
        """Run the model's generation in a thread."""
        try:
            input_ids = self._tokenizer.encode(prompt)
            input_tensor = torch.tensor([input_ids], device=self._device)

            with torch.no_grad():
                output_ids = self._model.generate(
                    input_tensor,
                    max_new_tokens=512,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                )

            new_ids = output_ids[0][len(input_ids):].tolist()
            return self._tokenizer.decode(new_ids)

        except Exception as exc:
            log.error("language_module.generate_error", error=str(exc))
            return f"Generation error: {exc}"

    async def stream(
        self,
        message: Any,
        episodic_context: list[dict],
        knowledge_context: list[dict],
        working_memory: list[dict],
    ) -> AsyncIterator[str]:
        """Stream response tokens one at a time."""
        if self._model is None or self._tokenizer is None:
            yield "Language model not initialised."
            return

        prompt = self._build_prompt(
            content=message.content,
            working_memory=working_memory,
            episodic_context=episodic_context,
            knowledge_context=knowledge_context,
        )

        input_ids = self._tokenizer.encode(prompt)
        input_tensor = torch.tensor([input_ids], device=self._device)
        generated = input_tensor.clone()

        with torch.no_grad():
            for _ in range(MAX_NEW_TOKENS):
                context = generated[:, -self._model.max_seq_len:]
                logits = self._model(context)
                next_token_logits = logits[:, -1, :] / TEMPERATURE

                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(
                    F.softmax(sorted_logits, dim=-1), dim=-1
                )
                sorted_indices_to_remove = (
                    cumulative_probs - F.softmax(sorted_logits, dim=-1) > TOP_P
                )
                sorted_logits[sorted_indices_to_remove] = float("-inf")
                next_token_logits_filtered = next_token_logits.scatter(
                    1, sorted_indices, sorted_logits
                )
                probs = F.softmax(next_token_logits_filtered, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated = torch.cat([generated, next_token], dim=1)

                token_id = next_token.item()
                if token_id == 2:  # EOS
                    break

                token_str = self._tokenizer.decode([token_id])
                if token_str:
                    yield token_str
                    await asyncio.sleep(0)  # Yield control to event loop
