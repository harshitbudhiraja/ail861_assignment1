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
    """Sinusoidal positional encoding from 'Attention is All You Need'."""
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (not a parameter, but part of state)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        return x + self.pe[:, :x.size(1), :]


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism."""
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear projections for Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask: Optional[torch.Tensor] = None):
        batch_size, seq_len, d_model = x.shape
        
        # Linear projections and reshape for multi-head attention
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply causal mask (for decoder)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)
        
        # Concatenate heads and apply output projection
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        output = self.W_o(attn_output)
        
        return output


class FeedForward(nn.Module):
    """Position-wise feed-forward network."""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class DecoderLayer(nn.Module):
    """Single decoder layer with masked self-attention and feed-forward network."""
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x, mask: Optional[torch.Tensor] = None):
        # Self-attention with residual connection and layer norm
        attn_output = self.self_attn(x, mask)
        x = self.norm1(x + self.dropout1(attn_output))
        
        # Feed-forward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))
        
        return x


class DecoderOnlyTransformer(nn.Module):
    """Complete decoder-only transformer architecture."""
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
        """Initialize parameters with Xavier uniform initialization."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create causal mask to prevent attending to future tokens."""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask == 0  # True for positions that can be attended to
        return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
    
    def create_padding_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Create mask for padded positions."""
        # x shape: (batch_size, seq_len)
        padding_mask = (x != self.pad_idx).unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, seq_len)
        return padding_mask
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the decoder-only transformer.
        
        Args:
            x: Input tensor of token indices, shape (batch_size, seq_len)
        
        Returns:
            Output logits, shape (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = x.shape
        
        causal_mask = self.create_causal_mask(seq_len, x.device)
        
        padding_mask = self.create_padding_mask(x)
        
        padding_mask = padding_mask.expand(-1, -1, seq_len, -1)
        
        mask = causal_mask & padding_mask  # Now broadcasts correctly
            
        x = self.token_embedding(x)
        
        if self.embedding_projection is not None:
            x = self.embedding_projection(x)
        
        x = x * math.sqrt(self.d_model) 
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Pass through decoder layers
        for layer in self.layers:
            x = layer(x, mask)
        
        x = self.norm(x)
        
        # Project to vocabulary
        logits = self.output_projection(x)
        
        return logits
    

    def generate_with_beam_search(self, start_tokens: torch.Tensor, max_length: int, beam_size: int = 5):
        pass

    def compute_loss(logits, targets, pad_idx):
    # Reshape for cross entropy
        logits = logits.reshape(-1, logits.size(-1))  # (batch*seq_len, vocab_size)
        targets = targets.reshape(-1)  # (batch*seq_len,)
        
        # Compute loss with ignore_index
        loss = F.cross_entropy(logits, targets, ignore_index=pad_idx)
        
        return loss


    def get_num_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def configure_optimizers(self, learning_rate: float = 3e-4, weight_decay: float = 0.01):
        """Configure AdamW optimizer with weight decay."""
        # Separate parameters that should and shouldn't have weight decay
        decay_params = []
        no_decay_params = []
        
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            # Don't apply weight decay to biases and layer norm parameters
            if 'bias' in name or 'norm' in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        optimizer = torch.optim.AdamW([
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ], lr=learning_rate)
        
        return optimizer


    def generate(
        self,
        start_tokens: torch.Tensor,
        max_length: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        eos_token_id: Optional[int] = None
    ) -> torch.Tensor:
        """
        Generate text autoregressively.
        
        Args:
            start_tokens: Initial tokens, shape (batch_size, start_len)
            max_length: Maximum length to generate
            temperature: Sampling temperature
            top_k: If set, only sample from top k tokens
            eos_token_id: Stop generation if this token is generated
        
        Returns:
            Generated token sequence, shape (batch_size, max_length)
        """
        self.eval()
        batch_size = start_tokens.shape[0]
        generated = start_tokens
        
        with torch.no_grad():
            for _ in range(max_length - start_tokens.shape[1]):
                # Get predictions for the current sequence
                logits = self.forward(generated)
                
                # Get logits for the last token
                next_token_logits = logits[:, -1, :] / temperature
                
                # Apply top-k filtering if specified
                if top_k is not None:
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                    next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                    next_token_logits.scatter_(1, top_k_indices, top_k_logits)
                
                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to generated sequence
                generated = torch.cat([generated, next_token], dim=1)
                
                # Check for EOS token
                if eos_token_id is not None and (next_token == eos_token_id).all():
                    break
        
        return generated


class FastTextEmbeddingLoader:
    """Helper class to load FastText embeddings for vocabulary."""
    
    def __init__(self, model_name: str = 'fasttext-wiki-news-subwords-300'):
        """
        Initialize FastText embedding loader.
        
        Args:
            model_name: Name of the FastText model to load from gensim
        """
        print(f"Loading FastText model: {model_name}")
        self.model = api.load(model_name)
        self.embedding_dim = self.model.vector_size
    
    def create_embedding_matrix(
        self,
        vocab: Dict[str, int],
        special_tokens: Dict[str, int]
    ) -> torch.Tensor:
        """
        Create embedding matrix for vocabulary using FastText.
        
        Args:
            vocab: Dictionary mapping words to indices
            special_tokens: Dictionary of special tokens (sos, eos, pad)
        
        Returns:
            Embedding matrix as torch.Tensor
        """
        vocab_size = len(vocab)
        embedding_matrix = torch.zeros(vocab_size, self.embedding_dim)
        
        found = 0
        for word, idx in vocab.items():
            if word in special_tokens.values():
                # Initialize special tokens randomly
                embedding_matrix[idx] = torch.randn(self.embedding_dim) * 0.01
            else:
                try:
                    embedding_matrix[idx] = torch.tensor(self.model[word])
                    found += 1
                except KeyError:
                    # Word not in FastText, initialize randomly
                    embedding_matrix[idx] = torch.randn(self.embedding_dim) * 0.01
        
        print(f"Found {found}/{vocab_size - len(special_tokens)} words in FastText")
        return embedding_matrix


# Example usage
if __name__ == "__main__":
    # Define special tokens
    SPECIAL_TOKENS = {
        '<pad>': 0,
        '<sos>': 1,
        '<eos>': 2,
        '<unk>': 3
    }
    
    # Example vocabulary (in practice, build from your dataset)
    vocab = {**SPECIAL_TOKENS, 'hello': 4, 'world': 5, 'transformer': 6}
    vocab_size = len(vocab)
    
    # Hyperparameters
    config = {
        'vocab_size': vocab_size,
        'd_model': 300,  # Match FastText dimension
        'num_layers': 6,
        'num_heads': 6,
        'd_ff': 1024,
        'max_seq_len': 512,
        'dropout': 0.1,
        'pad_idx': SPECIAL_TOKENS['<pad>']
    }
    
    
    # Option 2: Train embeddings from scratch
    model = DecoderOnlyTransformer(**config)
    
    # Example input
    batch_size = 2
    seq_len = 10
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Forward pass
    logits = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")
    
    # Generation example
    start_tokens = torch.tensor([[SPECIAL_TOKENS['<sos>']]])
    generated = model.generate(start_tokens, max_length=20, temperature=0.8, top_k=50)
    print(f"Generated sequence shape: {generated.shape}")