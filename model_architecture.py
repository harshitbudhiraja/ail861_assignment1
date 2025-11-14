import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional, Dict
import gensim.downloader as api

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.beta = nn.Parameter(torch.zeros(normalized_shape))
    
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x, offset: int = 0):
        seq_len = x.size(1)
        positions = torch.arange(offset, offset + seq_len, dtype=torch.float, device=x.device).unsqueeze(1)
        max_needed = offset + seq_len
        if max_needed <= self.max_len:
            pe = self.pe[:, offset:offset + seq_len, :].to(x.device)
        else:
            if offset < self.max_len:
                pre_computed = self.pe[:, offset:, :].to(x.device)
                additional_len = max_needed - self.max_len
                position = torch.arange(self.max_len, max_needed, dtype=torch.float, device=x.device).unsqueeze(1)
                div_term = torch.exp(torch.arange(0, self.d_model, 2, dtype=torch.float, device=x.device) * 
                                   (-math.log(10000.0) / self.d_model))
                additional_pe = torch.zeros(additional_len, self.d_model, device=x.device)
                additional_pe[:, 0::2] = torch.sin(position * div_term)
                additional_pe[:, 1::2] = torch.cos(position * div_term)
                pe = torch.cat([pre_computed, additional_pe.unsqueeze(0)], dim=1)
            else:
                div_term = torch.exp(torch.arange(0, self.d_model, 2, dtype=torch.float, device=x.device) * 
                                   (-math.log(10000.0) / self.d_model))
                pe = torch.zeros(seq_len, self.d_model, device=x.device)
                pe[:, 0::2] = torch.sin(positions * div_term)
                pe[:, 1::2] = torch.cos(positions * div_term)
                pe = pe.unsqueeze(0)
        return x + pe


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        x, 
        mask: Optional[torch.Tensor] = None, 
        return_attn: bool = False,
        past_kv: Optional[tuple] = None,
        use_cache: bool = False
    ):
        batch_size, seq_len, d_model = x.shape
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        if past_kv is not None:
            past_key, past_value = past_kv
            K = torch.cat([past_key, K], dim=-2)
            V = torch.cat([past_value, V], dim=-2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights_dropped = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights_dropped, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        output = self.W_o(attn_output)
        ret = [output]
        if return_attn:
            ret.append(attn_weights)
        if use_cache:
            ret.append((K, V))
        if len(ret) == 1:
            return ret[0]
        return tuple(ret)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(
        self, 
        x, 
        mask: Optional[torch.Tensor] = None, 
        return_attn: bool = False,
        past_kv: Optional[tuple] = None,
        use_cache: bool = False
    ):
        attn_kwargs = {
            'mask': mask,
            'return_attn': return_attn,
            'past_kv': past_kv,
            'use_cache': use_cache
        }
        attn_result = self.self_attn(x, **attn_kwargs)
        if use_cache and return_attn:
            attn_output, attn_weights, present_kv = attn_result
        elif use_cache:
            attn_output, present_kv = attn_result
            attn_weights = None
        elif return_attn:
            attn_output, attn_weights = attn_result
            present_kv = None
        else:
            attn_output = attn_result
            attn_weights = None
            present_kv = None
        x = self.norm1(x + self.dropout1(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))
        ret = [x]
        if return_attn:
            ret.append(attn_weights)
        if use_cache:
            ret.append(present_kv)
        if len(ret) == 1:
            return ret[0]
        return tuple(ret)


class DecoderOnlyTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        dropout: float = 0.1,
        fasttext_embeddings: Optional[torch.Tensor] = None,
        freeze_embeddings: bool = False,
        pad_idx: int = 0
    ):
        super().__init__()
        self.d_model = d_model
        self.pad_idx = pad_idx
        if fasttext_embeddings is not None:
            embedding_dim = fasttext_embeddings.shape[1]
            self.token_embedding = nn.Embedding.from_pretrained(
                fasttext_embeddings,
                freeze=freeze_embeddings,
                padding_idx=pad_idx
            )
            if embedding_dim != d_model:
                self.embedding_projection = nn.Linear(embedding_dim, d_model)
            else:
                self.embedding_projection = None
        else:
            self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
            self.embedding_projection = None
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_seq_len)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, vocab_size)
        self._init_parameters()
    
    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask == 0
        return mask.unsqueeze(0).unsqueeze(0)
    
    def create_padding_mask(self, x: torch.Tensor) -> torch.Tensor:
        padding_mask = (x != self.pad_idx).unsqueeze(1).unsqueeze(2)
        return padding_mask
    
    def forward(
        self, 
        x: torch.Tensor, 
        return_attn: bool = False,
        past_key_values: Optional[list] = None,
        use_cache: bool = False
    ) -> torch.Tensor:
        batch_size, seq_len = x.shape
        if past_key_values is not None and len(past_key_values) > 0:
            if seq_len == 1:
                mask = None
            else:
                causal_mask = self.create_causal_mask(seq_len, x.device)
                padding_mask = self.create_padding_mask(x)
                padding_mask = padding_mask.expand(-1, -1, seq_len, -1)
                mask = causal_mask & padding_mask
        else:
            causal_mask = self.create_causal_mask(seq_len, x.device)
            padding_mask = self.create_padding_mask(x)
            padding_mask = padding_mask.expand(-1, -1, seq_len, -1)
            mask = causal_mask & padding_mask
        x = self.token_embedding(x)
        if self.embedding_projection is not None:
            x = self.embedding_projection(x)
        x = x * math.sqrt(self.d_model)
        if past_key_values is not None and len(past_key_values) > 0:
            past_seq_len = past_key_values[0][0].shape[-2]
            x = self.pos_encoding(x, offset=past_seq_len)
        else:
            x = self.pos_encoding(x, offset=0)
        x = self.dropout(x)
        all_attn_weights = []
        present_key_values = [] if use_cache else None
        for layer_idx, layer in enumerate(self.layers):
            past_kv = past_key_values[layer_idx] if past_key_values is not None else None
            layer_kwargs = {
                'mask': mask,
                'return_attn': return_attn,
                'past_kv': past_kv,
                'use_cache': use_cache
            }
            layer_result = layer(x, **layer_kwargs)
            if use_cache and return_attn:
                x, attn_weights, present_kv = layer_result
                all_attn_weights.append(attn_weights)
                present_key_values.append(present_kv)
            elif use_cache:
                x, present_kv = layer_result
                present_key_values.append(present_kv)
            elif return_attn:
                x, attn_weights = layer_result
                all_attn_weights.append(attn_weights)
            else:
                x = layer_result
        x = self.norm(x)
        logits = self.output_projection(x)
        ret = [logits]
        if return_attn:
            ret.append(all_attn_weights)
        if use_cache:
            ret.append(present_key_values)
        if len(ret) == 1:
            return ret[0]
        return tuple(ret)
    

    def generate_with_beam_search(self, start_tokens: torch.Tensor, max_length: int, beam_size: int = 5):
        pass

    def compute_loss(logits, targets, pad_idx):
        logits = logits.reshape(-1, logits.size(-1))
        targets = targets.reshape(-1)
        loss = F.cross_entropy(logits, targets, ignore_index=pad_idx)
        return loss

    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def configure_optimizers(self, learning_rate: float = 3e-4, weight_decay: float = 0.01):
        decay_params = []
        no_decay_params = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if 'bias' in name or 'norm' in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        optimizer = torch.optim.AdamW([
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ], lr=learning_rate)
        return optimizer
    def _sample_generate(
        self,
        start_tokens,
        max_length,
        temperature,
        top_k,
        eos_token_id
    ):
        batch_size = start_tokens.shape[0]
        generated = start_tokens
        past_key_values = None
        with torch.no_grad():
            logits, past_key_values = self.forward(
                start_tokens, 
                use_cache=True
            )
            next_token_logits = logits[:, -1, :] / temperature
            if top_k is not None:
                top_k_logits, top_k_idx = torch.topk(next_token_logits, top_k)
                filtered = torch.full_like(next_token_logits, float("-inf"))
                filtered.scatter_(1, top_k_idx, top_k_logits)
                next_token_logits = filtered
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            generated = torch.cat([generated, next_token], dim=1)
            if eos_token_id is not None and (next_token == eos_token_id).all():
                return generated
            for _ in range(max_length - start_tokens.shape[1] - 1):
                logits, past_key_values = self.forward(
                    next_token,
                    past_key_values=past_key_values,
                    use_cache=True
                )
                next_token_logits = logits[:, -1, :] / temperature
                if top_k is not None:
                    top_k_logits, top_k_idx = torch.topk(next_token_logits, top_k)
                    filtered = torch.full_like(next_token_logits, float("-inf"))
                    filtered.scatter_(1, top_k_idx, top_k_logits)
                    next_token_logits = filtered
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                generated = torch.cat([generated, next_token], dim=1)
                if eos_token_id is not None and (next_token == eos_token_id).all():
                    break
        return generated

    def generate(
    self,
    start_tokens: torch.Tensor,
    max_length: int,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    eos_token_id: Optional[int] = None,
    beam_search: bool = False,
    beam_size: int = 3,
) -> torch.Tensor:

        self.eval()
        B = start_tokens.size(0)
        device = start_tokens.device
        if not beam_search:
            return self._sample_generate(
                start_tokens, max_length, temperature, top_k, eos_token_id
            )
        with torch.no_grad():
            beams = [ (start_tokens, torch.zeros(B, device=device)) ]
            for _ in range(max_length - start_tokens.size(1)):
                new_beams = []
                for seq, seq_logprob in beams:
                    logits = self.forward(seq)[:, -1, :]
                    logprobs = F.log_softmax(logits, dim=-1)
                    top_logprobs, top_idx = torch.topk(logprobs, beam_size, dim=-1)
                    for b in range(beam_size):
                        next_token = top_idx[:, b].unsqueeze(-1)
                        next_score = seq_logprob + top_logprobs[:, b]
                        new_seq = torch.cat([seq, next_token], dim=1)
                        new_beams.append((new_seq, next_score))
                all_scores = torch.stack([score for (_, score) in new_beams])
                best_indices = torch.topk(all_scores, beam_size, dim=0).indices
                beams = []
                for k in range(beam_size):
                    idx = best_indices[k]
                    beams.append(new_beams[idx])
                if eos_token_id is not None:
                    all_end = True
                    for seq, _ in beams:
                        if not (seq[:, -1] == eos_token_id).all():
                            all_end = False
                    if all_end:
                        break
            best_seq, _ = beams[0]
            return best_seq

class FastTextEmbeddingLoader:
    def __init__(self, model_name: str = 'fasttext-wiki-news-subwords-300'):
        print(f"Loading FastText model: {model_name}")
        self.model = api.load(model_name)
        self.embedding_dim = self.model.vector_size
    
    def create_embedding_matrix(
        self,
        vocab: Dict[str, int],
        special_tokens: Dict[str, int]
    ) -> torch.Tensor:
        vocab_size = len(vocab)
        embedding_matrix = torch.zeros(vocab_size, self.embedding_dim)
        found = 0
        for word, idx in vocab.items():
            if word in special_tokens.values():
                embedding_matrix[idx] = torch.randn(self.embedding_dim) * 0.01
            else:
                try:
                    embedding_matrix[idx] = torch.tensor(self.model[word])
                    found += 1
                except KeyError:
                    embedding_matrix[idx] = torch.randn(self.embedding_dim) * 0.01
        print(f"Found {found}/{vocab_size - len(special_tokens)} words in FastText")
        return embedding_matrix

if __name__ == "__main__":
    SPECIAL_TOKENS = {
        '<pad>': 0,
        '<sos>': 1,
        '<eos>': 2,
        '<unk>': 3
    }
    vocab = {**SPECIAL_TOKENS, 'hello': 4, 'world': 5, 'transformer': 6}
    vocab_size = len(vocab)
    config = {
        'vocab_size': vocab_size,
        'd_model': 300,
        'num_layers': 6,
        'num_heads': 6,
        'd_ff': 1024,
        'max_seq_len': 512,
        'dropout': 0.1,
        'pad_idx': SPECIAL_TOKENS['<pad>']
    }
    model = DecoderOnlyTransformer(**config)
    batch_size = 2
    seq_len = 10
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    logits = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")
    start_tokens = torch.tensor([[SPECIAL_TOKENS['<sos>']]])
    generated = model.generate(start_tokens, max_length=20, temperature=0.8, top_k=50)
    print(f"Generated sequence shape: {generated.shape}")