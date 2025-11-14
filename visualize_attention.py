import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Tuple, Optional

def visualize_attention_weights(
    attention_weights: torch.Tensor,
    tokens: List[str],
    layer_idx: int,
    head_idx: int,
    save_path: Optional[str] = None,
    title: Optional[str] = None
):
    """
    Visualize attention weights for a specific layer and head.
    
    Args:
        attention_weights: Attention weights tensor of shape (batch, num_heads, seq_len, seq_len)
        tokens: List of token strings corresponding to the sequence
        layer_idx: Index of the layer
        head_idx: Index of the attention head
        save_path: Path to save the figure
        title: Title for the plot
    """
    # Extract attention weights for specific layer and head
    # attention_weights is a list of tensors, one per layer
    # Each tensor has shape (batch, num_heads, seq_len, seq_len)
    if isinstance(attention_weights, list):
        attn = attention_weights[layer_idx][0, head_idx].cpu().numpy()  # (seq_len, seq_len)
    else:
        attn = attention_weights[0, head_idx].cpu().numpy()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create heatmap
    sns.heatmap(
        attn,
        xticklabels=tokens,
        yticklabels=tokens,
        cmap='Blues',
        annot=False,
        fmt='.2f',
        cbar=True,
        ax=ax,
        square=True
    )
    
    # Set labels and title
    ax.set_xlabel('Key Position', fontsize=12)
    ax.set_ylabel('Query Position', fontsize=12)
    if title is None:
        title = f'Attention Weights - Layer {layer_idx}, Head {head_idx}'
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Rotate labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Attention visualization saved to {save_path}")
    
    return fig

def visualize_all_heads(
    attention_weights: List[torch.Tensor],
    tokens: List[str],
    layer_idx: int,
    save_path: Optional[str] = None
):
    """
    Visualize all attention heads for a specific layer.
    
    Args:
        attention_weights: List of attention weight tensors, one per layer
        tokens: List of token strings
        layer_idx: Index of the layer to visualize
        save_path: Path to save the figure
    """
    attn_layer = attention_weights[layer_idx][0]  # (num_heads, seq_len, seq_len)
    num_heads = attn_layer.shape[0]
    
    # Create subplots for all heads
    fig, axes = plt.subplots(1, num_heads, figsize=(5 * num_heads, 5))
    if num_heads == 1:
        axes = [axes]
    
    for head_idx in range(num_heads):
        attn = attn_layer[head_idx].cpu().numpy()
        
        sns.heatmap(
            attn,
            xticklabels=tokens if head_idx == 0 else False,
            yticklabels=tokens,
            cmap='Blues',
            annot=False,
            cbar=True,
            ax=axes[head_idx],
            square=True
        )
        
        axes[head_idx].set_title(f'Head {head_idx}', fontsize=12, fontweight='bold')
        if head_idx == 0:
            axes[head_idx].set_ylabel('Query Position', fontsize=10)
        axes[head_idx].set_xlabel('Key Position', fontsize=10)
        plt.setp(axes[head_idx].get_xticklabels(), rotation=45, ha='right')
    
    plt.suptitle(f'All Attention Heads - Layer {layer_idx}', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"All heads visualization saved to {save_path}")
    
    return fig

def visualize_attention_patterns(
    model,
    processor,
    text: str,
    device: torch.device,
    save_dir: str = "attention_visualizations"
):
    """
    Generate attention visualizations for a given text input.
    
    Args:
        model: The transformer model
        processor: Text processor for tokenization
        text: Input text string
        device: Device to run inference on
        save_dir: Directory to save visualizations
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    # Tokenize text
    token_ids = processor.text_to_ids(text, add_special_tokens=True)
    tokens = [processor.idx2word.get(idx, '<unk>') for idx in token_ids]
    
    # Convert to tensor
    input_tensor = torch.tensor([token_ids], dtype=torch.long).to(device)
    
    # Get attention weights
    model.eval()
    with torch.no_grad():
        _, attention_weights = model(input_tensor, return_attn=True)
    
    # Visualize for each layer
    num_layers = len(attention_weights)
    num_heads = attention_weights[0].shape[1]
    
    print(f"Visualizing attention for {num_layers} layers and {num_heads} heads per layer")
    print(f"Text: {text}")
    print(f"Tokens: {tokens}")
    
    # Visualize all heads for each layer
    for layer_idx in range(num_layers):
        save_path = os.path.join(save_dir, f"layer_{layer_idx}_all_heads.png")
        visualize_all_heads(attention_weights, tokens, layer_idx, save_path)
        plt.close()
    
    # Visualize individual heads (first and last layer, first and last head)
    for layer_idx in [0, num_layers - 1]:
        for head_idx in [0, num_heads - 1]:
            save_path = os.path.join(save_dir, f"layer_{layer_idx}_head_{head_idx}.png")
            visualize_attention_weights(
                attention_weights,
                tokens,
                layer_idx,
                head_idx,
                save_path,
                title=f'Layer {layer_idx}, Head {head_idx}'
            )
            plt.close()
    
    print(f"All visualizations saved to {save_dir}/")

