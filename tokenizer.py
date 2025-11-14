import numpy as np
import torch
from gensim.models import KeyedVectors
import re
import os
import nltk
from nltk.tokenize import word_tokenize
from collections import Counter
from typing import List, Tuple, Dict
import pickle

nltk.download('punkt')
nltk.download('punkt_tab')


class TextProcessor:
    def __init__(self, fasttext_path='cc.en.300.vec', seq_len=128, min_freq=2):
        """
        Initialize text processor with FastText embeddings.
        
        Args:
            fasttext_path: Path to FastText word vectors file
            seq_len: Maximum sequence length
            min_freq: Minimum word frequency to include in vocabulary
        """
        self.seq_len = seq_len
        self.min_freq = min_freq
        
        # Special tokens
        self.special_tokens = {
            '<pad>': 0,
            '<sos>': 1,
            '<eos>': 2,
            '<unk>': 3
        }
        
        # Will be built from training data
        self.word2idx = self.special_tokens.copy()
        self.idx2word = {v: k for k, v in self.special_tokens.items()}
        self.vocab_size = len(self.special_tokens)
        
        # FastText model (loaded on demand)
        self.fasttext_path = fasttext_path
        self.fasttext = None
        self.embedding_dim = 300
        
    def build_vocab(self, stories_file: str, max_vocab_size: int = 50000):
        """
        Build vocabulary from training data.
        
        Args:
            stories_file: Path to text file with stories
            max_vocab_size: Maximum vocabulary size (excluding special tokens)
        """
        print("Building vocabulary from training data...")
        
        # Count word frequencies
        word_freq = Counter()
        with open(stories_file, 'r', encoding='utf-8') as f:
            for line in f:
                tokens = self._tokenize(line)
                word_freq.update(tokens)
        
        # Filter by frequency and limit vocab size
        valid_words = [
            word for word, freq in word_freq.most_common()
            if freq >= self.min_freq
        ][:max_vocab_size]
        
        # Build word2idx mapping (special tokens already added)
        for word in valid_words:
            if word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word
        
        self.vocab_size = len(self.word2idx)
        print(f"Vocabulary built: {self.vocab_size} tokens (including {len(self.special_tokens)} special tokens)")
        print(f"Most common words: {valid_words[:10]}")
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        tokens = word_tokenize(text)
        return tokens
    
    def text_to_ids(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Convert text to sequence of token IDs.
        
        Args:
            text: Input text string
            add_special_tokens: Whether to add <sos> and <eos> tokens
        
        Returns:
            List of token IDs
        """
        tokens = self._tokenize(text)
        
        # Convert tokens to IDs
        token_ids = []
        
        if add_special_tokens:
            token_ids.append(self.special_tokens['<sos>'])
        
        for token in tokens:
            if token in self.word2idx:
                token_ids.append(self.word2idx[token])
            else:
                token_ids.append(self.special_tokens['<unk>'])
        
        if add_special_tokens:
            token_ids.append(self.special_tokens['<eos>'])
        
        return token_ids
    
    def ids_to_text(self, token_ids: List[int]) -> str:
        """Convert token IDs back to text."""
        tokens = [self.idx2word.get(idx, '<unk>') for idx in token_ids]
        # Remove special tokens for display
        tokens = [t for t in tokens if t not in self.special_tokens]
        return ' '.join(tokens)
    
    def pad_sequence(self, token_ids: List[int], max_len: int = None) -> List[int]:
        """
        Pad or truncate sequence to max_len.
        
        Args:
            token_ids: List of token IDs
            max_len: Maximum length (defaults to self.seq_len)
        
        Returns:
            Padded/truncated list of token IDs
        """
        if max_len is None:
            max_len = self.seq_len
        
        # Truncate if too long
        if len(token_ids) > max_len:
            token_ids = token_ids[:max_len]
        
        # Pad if too short
        while len(token_ids) < max_len:
            token_ids.append(self.special_tokens['<pad>'])
        
        return token_ids
    
    def prepare_training_data(self, stories_file: str, batch_size: int = 32):
        """
        Process stories into batches of (input_ids, target_ids) for training.
        
        The transformer will:
        - Take input_ids and predict next token at each position
        - Compare predictions with target_ids using cross-entropy loss
        
        Args:
            stories_file: Path to text file with stories
            batch_size: Batch size
        
        Yields:
            Tuple of (input_ids, target_ids) arrays
            - input_ids: shape (batch_size, seq_len)
            - target_ids: shape (batch_size, seq_len)
        """
        with open(stories_file, 'r', encoding='utf-8') as f:
            stories = f.readlines()
        
        input_batch = []
        target_batch = []
        
        for story in stories:
            # Convert to token IDs
            token_ids = self.text_to_ids(story, add_special_tokens=True)
            
            # Skip if too short (need at least 2 tokens for input/target pair)
            if len(token_ids) < 2:
                continue
            
            # Pad sequence
            token_ids = self.pad_sequence(token_ids)
            
            # Create input and target sequences
            # Input: [<sos>, w1, w2, w3, ..., wN-1]
            # Target: [w1, w2, w3, ..., wN-1, <eos>]
            input_ids = token_ids[:-1]  # All tokens except last
            target_ids = token_ids[1:]  # All tokens except first (shifted by 1)
            
            input_batch.append(input_ids)
            target_batch.append(target_ids)
            
            # Yield batch when ready
            if len(input_batch) >= batch_size:
                yield np.array(input_batch, dtype=np.int64), np.array(target_batch, dtype=np.int64)
                input_batch = []
                target_batch = []
        
        # Yield remaining data
        if input_batch:
            yield np.array(input_batch, dtype=np.int64), np.array(target_batch, dtype=np.int64)
    
    def create_fasttext_embedding_matrix(self) -> torch.Tensor:
        """
        Create embedding matrix using FastText pre-trained vectors.
        
        Returns:
            Embedding matrix as torch.Tensor, shape (vocab_size, embedding_dim)
        """
        print(f"Loading FastText vectors from {self.fasttext_path}...")
        self.fasttext = KeyedVectors.load_word2vec_format(self.fasttext_path)
        
        embedding_matrix = np.zeros((self.vocab_size, self.embedding_dim))
        found = 0
        
        for word, idx in self.word2idx.items():
            if word in self.special_tokens:
                # Initialize special tokens randomly
                embedding_matrix[idx] = np.random.randn(self.embedding_dim) * 0.01
            else:
                try:
                    embedding_matrix[idx] = self.fasttext[word]
                    found += 1
                except KeyError:
                    # Word not in FastText, initialize randomly
                    embedding_matrix[idx] = np.random.randn(self.embedding_dim) * 0.01
        
        print(f"Found {found}/{self.vocab_size - len(self.special_tokens)} words in FastText")
        
        return torch.FloatTensor(embedding_matrix)
    
    def save_vocab(self, path: str):
        """Save vocabulary to file."""
        vocab_data = {
            'word2idx': self.word2idx,
            'idx2word': self.idx2word,
            'vocab_size': self.vocab_size,
            'special_tokens': self.special_tokens
        }
        with open(path, 'wb') as f:
            pickle.dump(vocab_data, f)
        print(f"Vocabulary saved to {path}")
    
    def load_vocab(self, path: str):
        """Load vocabulary from file."""
        with open(path, 'rb') as f:
            vocab_data = pickle.load(f)
        
        self.word2idx = vocab_data['word2idx']
        self.idx2word = vocab_data['idx2word']
        self.vocab_size = vocab_data['vocab_size']
        self.special_tokens = vocab_data['special_tokens']
        print(f"Vocabulary loaded from {path}: {self.vocab_size} tokens")


def prepare_data_pipeline(stories_file: str, cache_dir: str = "cache", 
                         batch_size: int = 32, use_fasttext: bool = True,
                         fasttext_path: str = 'cc.en.300.vec'):
    """
    Complete data preparation pipeline.
    
    Args:
        stories_file: Path to training data
        cache_dir: Directory to cache processed data
        batch_size: Batch size
        use_fasttext: Whether to use FastText embeddings
        fasttext_path: Path to FastText vectors
    
    Returns:
        Tuple of (processor, first_batch_input, first_batch_target, embedding_matrix)
    """
    os.makedirs(cache_dir, exist_ok=True)
    
    vocab_path = os.path.join(cache_dir, "vocab.pkl")
    x_cache_path = os.path.join(cache_dir, "X_batch.npy")
    y_cache_path = os.path.join(cache_dir, "y_batch.npy")
    embed_cache_path = os.path.join(cache_dir, "embeddings.pt")
    
    processor = TextProcessor(fasttext_path=fasttext_path, seq_len=128)
    
    if os.path.exists(vocab_path):
        print("Loading vocabulary from cache...")
        processor.load_vocab(vocab_path)
    else:
        print("Building vocabulary...")
        processor.build_vocab(stories_file, max_vocab_size=30000)
        processor.save_vocab(vocab_path)
    
    # Load or create first batch
    if os.path.exists(x_cache_path) and os.path.exists(y_cache_path):
        print("Loading batches from cache...")
        X_batch = np.load(x_cache_path)
        y_batch = np.load(y_cache_path)
    else:
        print("Creating batches...")
        data_generator = processor.prepare_training_data(stories_file, batch_size=batch_size)
        X_batch, y_batch = next(data_generator)
        np.save(x_cache_path, X_batch)
        np.save(y_cache_path, y_batch)
        print(f"Saved batches to {cache_dir}/")
    
    # Create embedding matrix if using FastText
    embedding_matrix = None
    if use_fasttext:
        if os.path.exists(embed_cache_path):
            print("Loading embedding matrix from cache...")
            embedding_matrix = torch.load(embed_cache_path)
        else:
            print("Creating FastText embedding matrix...")
            embedding_matrix = processor.create_fasttext_embedding_matrix()
            torch.save(embedding_matrix, embed_cache_path)
            print(f"Saved embeddings to {cache_dir}/")
    
    
    return processor, X_batch, y_batch, embedding_matrix


if __name__ == "__main__":

    processor, X_batch, y_batch, embeddings = prepare_data_pipeline(
        stories_file="data/tinystories.txt",
        cache_dir="cache",
        batch_size=32,
        use_fasttext=True,  # Set to False to train embeddings from scratch
        fasttext_path="cc.en.300.vec"
    )
    
    print("\n" + "="*50)
    print("Ready to train!")
    print("="*50)
    print(f"Use these in your transformer:")
    print(f"  vocab_size = {processor.vocab_size}")
    print(f"  pad_idx = {processor.special_tokens['<pad>']}")
    print(f"  Input shape: {X_batch.shape}  (batch_size, seq_len-1)")
    print(f"  Target shape: {y_batch.shape}  (batch_size, seq_len-1)")
    if embeddings is not None:
        print(f"  Embeddings shape: {embeddings.shape}  (vocab_size, embedding_dim)")