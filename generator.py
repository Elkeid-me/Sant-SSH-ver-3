# -*- coding:utf-8 -*-

import math

import torch
from torch import nn
from torch.nn import functional

from config import Config
from vocab import Vocab


class _TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size: int, device: str):
        super(_TokenEmbedding, self).__init__()
        self._embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_size, device=device)
        self._sqrt_emb_size = math.sqrt(emb_size)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self._embedding(tokens) * self._sqrt_emb_size


class _PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, device: str, dropout=0.1, max_len=65):
        super(_PositionalEncoding, self).__init__()
        self._dropout = nn.Dropout(p=dropout)

        pe = torch.zeros((max_len, d_model), device=device)
        position: torch.Tensor = torch.arange(0, max_len, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, device=device).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0), :]
        return self._dropout(x)


class Generator(nn.Module):
    def __init__(self, vocab_size: int,
                 d_model: int,
                 nhead: int,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 dim_feedforward: int,
                 dropout: float, device: str):
        super(Generator, self).__init__()

        self._token_embedding = _TokenEmbedding(vocab_size=vocab_size, emb_size=d_model, device=device)
        self._position_encoding = _PositionalEncoding(d_model=d_model, dropout=dropout, device=device)

        self._transformer = nn.Transformer(d_model=d_model,
                                           nhead=nhead,
                                           num_encoder_layers=num_encoder_layers,
                                           num_decoder_layers=num_decoder_layers,
                                           dim_feedforward=dim_feedforward,
                                           dropout=dropout, device=device,
                                           activation=functional.gelu)

        self._classification = nn.Linear(in_features=d_model,
                                         out_features=vocab_size,
                                         device=device)

    def forward(self,
                src=None, tgt=None,
                src_mask=None, tgt_mask=None,
                memory_mask=None,
                src_key_padding_mask=None,
                tgt_key_padding_mask=None,
                memory_key_padding_mask=None) -> torch.Tensor:
        src_embed = self._token_embedding(src)
        src_embed = self._position_encoding(src_embed)
        tgt_embed = self._token_embedding(tgt)
        tgt_embed = self._position_encoding(tgt_embed)
        trans_out = self._transformer(src=src_embed, tgt=tgt_embed,
                                      src_mask=src_mask, tgt_mask=tgt_mask,
                                      memory_mask=memory_mask,
                                      src_key_padding_mask=src_key_padding_mask,
                                      tgt_key_padding_mask=tgt_key_padding_mask,
                                      memory_key_padding_mask=memory_key_padding_mask)
        out = self._classification(trans_out)
        return out

    def _encode(self, src: torch.Tensor) -> torch.Tensor:
        src_embed = self._token_embedding(src)
        src_embed = self._position_encoding(src_embed)
        memory = self._transformer.encoder(src_embed)
        return memory

    def _decode(self, tgt: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        tgt_embed = self._token_embedding(tgt)
        tgt_embed = self._position_encoding(tgt_embed)
        out = self._transformer.decoder(tgt_embed, memory=memory)
        return out

    def greedy_decode(self, vocab_: Vocab, config_: Config, src_: str) -> str:
        src_tensor = \
            torch.tensor([vocab_.char_to_int(char) for char in src_], device=config_.device).reshape(-1, 1)
        start_symbol_index: int = vocab_.BOS_INDEX
        max_len: int = len(src_)
        memory = self._encode(src_tensor)
        out_tensor = torch.tensor([[start_symbol_index]], device=config_.device)
        next_word, last_word = 0, 0
        for i in range(0, max_len):
            if src_[i] == "，":  # 对逗号早期处理
                next_word = vocab_.char_to_int("，")
            # elif i > 0 and src_[i] == src_[i - 1]:  # 叠字
            #     pass
            else:
                out = self._decode(out_tensor, memory)
                out = self._classification(out[-1, :])
                out[:, vocab_.EOS_INDEX] *= 0
                if src_[i] not in vocab_.marks:  # 上联对应位置不是标点
                    for mark in vocab_.marks:
                        out[:, vocab_.char_to_int(mark)] *= 0
                    out[:, vocab_.char_to_int(src_[i])] *= 0  # 保证与上联不同
                    if i > 0 and src_[i] != src_[i - 1]:  # 上联不是叠字
                        out[:, last_word] *= 0
                else:
                    for mark in vocab_.marks:
                        out[:, vocab_.char_to_int(mark)] += 100
                _, next_word = torch.max(out, dim=1)
                next_word = next_word.item()

            last_word = next_word
            out_tensor = torch.cat([out_tensor, torch.tensor([[next_word]], device=config_.device)], dim=0)

        out_list: list[str] = [vocab_.int_to_char(index) for index in out_tensor]
        return "".join(out_list[1:])
