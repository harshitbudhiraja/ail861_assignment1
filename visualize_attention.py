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
    if isinstance(attention_weights, list):
        attn = attention_weights[layer_idx][0, head_idx].cpu().numpy()
    else:
        attn = attention_weights[0, head_idx].cpu().numpy()
    fig, ax = plt.subplots(figsize=(12, 10))
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
    ax.set_xlabel('Key Position', fontsize=12)
    ax.set_ylabel('Query Position', fontsize=12)
    if title is None:
        title = f'Attention Weights - Layer {layer_idx}, Head {head_idx}'
    ax.set_title(title, fontsize=14, fontweight='bold')
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
    attn_layer = attention_weights[layer_idx][0]
    num_heads = attn_layer.shape[0]
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
    import os
    os.makedirs(save_dir, exist_ok=True)
    token_ids = processor.text_to_ids(text, add_special_tokens=True)
    tokens = [processor.idx2word.get(idx, '<unk>') for idx in token_ids]
    input_tensor = torch.tensor([token_ids], dtype=torch.long).to(device)
    model.eval()
    with torch.no_grad():
        _, attention_weights = model(input_tensor, return_attn=True)
    num_layers = len(attention_weights)
    num_heads = attention_weights[0].shape[1]
    print(f"Visualizing attention for {num_layers} layers and {num_heads} heads per layer")
    print(f"Text: {text}")
    print(f"Tokens: {tokens}")
    for layer_idx in range(num_layers):
        save_path = os.path.join(save_dir, f"layer_{layer_idx}_all_heads.png")
        visualize_all_heads(attention_weights, tokens, layer_idx, save_path)
        plt.close()
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

