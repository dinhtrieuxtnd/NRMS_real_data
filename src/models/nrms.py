from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class NRMSConfig:
    embedding_dim: int = 300
    num_attention_heads: int = 16
    attention_head_dim: int = 16
    additive_attention_dim: int = 200
    dropout: float = 0.2
    freeze_embedding: bool = False

    @property
    def news_dim(self) -> int:
        return self.num_attention_heads * self.attention_head_dim


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, input_dim: int, num_heads: int, head_dim: int) -> None:
        super().__init__()
        if input_dim <= 0 or num_heads <= 0 or head_dim <= 0:
            raise ValueError("attention dimensions must be positive")

        self.num_heads = num_heads
        self.head_dim = head_dim
        self.output_dim = num_heads * head_dim
        self.query_projection = nn.Linear(input_dim, self.output_dim)
        self.key_projection = nn.Linear(input_dim, self.output_dim)
        self.value_projection = nn.Linear(input_dim, self.output_dim)
        self.scale = head_dim**-0.5

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if inputs.shape[:-1] != mask.shape:
            raise ValueError("mask shape must match the input sequence shape")

        sequence_length = inputs.shape[-2]
        batch_shape = inputs.shape[:-2]
        projected_shape = (*batch_shape, sequence_length, self.num_heads, self.head_dim)

        queries = self.query_projection(inputs).reshape(projected_shape).transpose(-3, -2)
        keys = self.key_projection(inputs).reshape(projected_shape).transpose(-3, -2)
        values = self.value_projection(inputs).reshape(projected_shape).transpose(-3, -2)

        scores = torch.matmul(queries, keys.transpose(-2, -1)) * self.scale
        key_mask = mask.to(torch.bool).unsqueeze(-2).unsqueeze(-2)
        attention_weights = _masked_softmax(scores, key_mask, dim=-1)
        contexts = torch.matmul(attention_weights, values)
        contexts = contexts.transpose(-3, -2).reshape(
            *batch_shape, sequence_length, self.output_dim
        )
        return contexts * mask.unsqueeze(-1).to(contexts.dtype)


class AdditiveAttention(nn.Module):
    def __init__(self, input_dim: int, attention_dim: int) -> None:
        super().__init__()
        if input_dim <= 0 or attention_dim <= 0:
            raise ValueError("attention dimensions must be positive")

        self.projection = nn.Linear(input_dim, attention_dim)
        self.query = nn.Parameter(torch.empty(attention_dim))
        nn.init.normal_(self.query, mean=0.0, std=0.1)

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if inputs.shape[:-1] != mask.shape:
            raise ValueError("mask shape must match the input sequence shape")

        scores = torch.matmul(torch.tanh(self.projection(inputs)), self.query)
        attention_weights = _masked_softmax(
            scores, mask.to(torch.bool), dim=-1
        )
        return torch.sum(attention_weights.unsqueeze(-1) * inputs, dim=-2)


class NewsEncoder(nn.Module):
    def __init__(
        self,
        embedding_matrix: np.ndarray | torch.Tensor,
        config: NRMSConfig,
    ) -> None:
        super().__init__()
        embedding_tensor = torch.as_tensor(
            embedding_matrix, dtype=torch.float32
        ).clone()
        if embedding_tensor.ndim != 2:
            raise ValueError("embedding_matrix must be two-dimensional")
        if embedding_tensor.shape[1] != config.embedding_dim:
            raise ValueError(
                "embedding_matrix width must equal config.embedding_dim"
            )

        self.embedding = nn.Embedding.from_pretrained(
            embedding_tensor,
            freeze=config.freeze_embedding,
            padding_idx=0,
        )
        self.dropout = nn.Dropout(config.dropout)
        self.self_attention = MultiHeadSelfAttention(
            config.embedding_dim,
            config.num_attention_heads,
            config.attention_head_dim,
        )
        self.additive_attention = AdditiveAttention(
            config.news_dim,
            config.additive_attention_dim,
        )

    def forward(self, title_tokens: torch.Tensor) -> torch.Tensor:
        token_mask = title_tokens.ne(0)
        embeddings = self.dropout(self.embedding(title_tokens))
        contextual_words = self.self_attention(embeddings, token_mask)
        return self.additive_attention(contextual_words, token_mask)


class UserEncoder(nn.Module):
    def __init__(self, config: NRMSConfig) -> None:
        super().__init__()
        self.self_attention = MultiHeadSelfAttention(
            config.news_dim,
            config.num_attention_heads,
            config.attention_head_dim,
        )
        self.additive_attention = AdditiveAttention(
            config.news_dim,
            config.additive_attention_dim,
        )

    def forward(
        self,
        clicked_news: torch.Tensor,
        history_mask: torch.Tensor,
    ) -> torch.Tensor:
        contextual_news = self.self_attention(clicked_news, history_mask)
        return self.additive_attention(contextual_news, history_mask)


class NRMS(nn.Module):
    def __init__(
        self,
        embedding_matrix: np.ndarray | torch.Tensor,
        config: NRMSConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or NRMSConfig()
        self.news_encoder = NewsEncoder(embedding_matrix, self.config)
        self.user_encoder = UserEncoder(self.config)

    def forward(
        self,
        history: torch.Tensor,
        history_mask: torch.Tensor,
        candidates: torch.Tensor,
        candidate_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        clicked_news = self.news_encoder(history)
        candidate_news = self.news_encoder(candidates)
        return self.score_news_vectors(
            clicked_news,
            history_mask,
            candidate_news,
            candidate_mask,
        )

    def score_news_vectors(
        self,
        clicked_news: torch.Tensor,
        history_mask: torch.Tensor,
        candidate_news: torch.Tensor,
        candidate_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if clicked_news.shape[:-1] != history_mask.shape:
            raise ValueError("history_mask shape must match clicked_news")
        if clicked_news.shape[-1] != self.config.news_dim:
            raise ValueError("clicked_news width must equal config.news_dim")
        if candidate_news.shape[-1] != self.config.news_dim:
            raise ValueError("candidate_news width must equal config.news_dim")
        user_vector = self.user_encoder(clicked_news, history_mask)
        logits = torch.einsum("bd,bcd->bc", user_vector, candidate_news)
        if candidate_mask is not None:
            if candidate_mask.shape != logits.shape:
                raise ValueError("candidate_mask shape must match the logits shape")
            logits = logits.masked_fill(
                ~candidate_mask.to(torch.bool),
                torch.finfo(logits.dtype).min,
            )
        return logits


def _masked_softmax(
    inputs: torch.Tensor,
    mask: torch.Tensor,
    *,
    dim: int,
) -> torch.Tensor:
    mask = mask.to(torch.bool)
    masked_inputs = inputs.masked_fill(~mask, torch.finfo(inputs.dtype).min)
    probabilities = torch.softmax(masked_inputs, dim=dim) * mask.to(inputs.dtype)
    normalizer = probabilities.sum(dim=dim, keepdim=True)
    return probabilities / normalizer.clamp_min(torch.finfo(inputs.dtype).eps)