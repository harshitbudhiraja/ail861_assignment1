import torch
import torch.nn.functional as F
import wandb
import yaml
from tokenizer import prepare_data_pipeline
from model_architecture import DecoderOnlyTransformer
from evaluate_model import ModelEvaluator
from tqdm import tqdm
import numpy as np

# Load hyperparameters from YAML config
with open("configs/hyperparams.yaml", "r") as f:
    config = yaml.safe_load(f)
    context_length = config.get("context_length", 64)
    num_of_layers = config.get("num_of_layers", 3)
    num_of_heads = config.get("num_of_heads", 8)

wandb.init(project="transformer-training", name="decoder-only")
train_file = "data/tinystories_train.txt"
val_file = "data/tinystories_val.txt"

# Determine device early
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")

processor, X_batch, y_batch, embeddings = prepare_data_pipeline(
    stories_file=train_file,
    cache_dir="cache",
    batch_size=32,
    use_fasttext=False,
    fasttext_path="cc.en.300.vec",
    device=str(device)  # Pass device to pipeline
)
print(f"Vocab size: {processor.vocab_size}")

model = DecoderOnlyTransformer(
    vocab_size=processor.vocab_size,
    d_model=256, 
    num_layers=num_of_layers,
    num_heads=num_of_heads,
    d_ff=512,
    max_seq_len=context_length,
    dropout=0.1,
    fasttext_embeddings=embeddings,
    freeze_embeddings=False,  # Allow fine-tuning
    pad_idx=processor.special_tokens['<pad>']
)

num_params = model.get_num_parameters()
print(f"Model parameters: {num_params:,}")
wandb.log({"num_parameters": num_params})

# Move model to device
model = model.to(device)
print(f"Model moved to device: {device}")
optimizer = model.configure_optimizers(learning_rate=3e-4)
print(f"Optimizer created")
def train():
    model.train()
    # Track perplexity over training
    perplexity_history = []
    batch_indices = []
    
    for epoch in range(3):
        # Get batches
        data_generator = processor.prepare_training_data(train_file, batch_size=1024, use_tqdm=True)
        print(f"Data generator created")
        total_loss = 0
        num_batches = 0
        
        for batch_idx, (X_batch, y_batch) in enumerate(data_generator):
            # Move to device (convert numpy to tensor and move in one step)
            X_batch = torch.from_numpy(X_batch).long().to(device, non_blocking=True)
            y_batch = torch.from_numpy(y_batch).long().to(device, non_blocking=True)
            
            logits = model(X_batch)
            
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y_batch.reshape(-1),
                ignore_index=processor.special_tokens["<pad>"]
            )
            
            # Calculate perplexity (exp of loss)
            perplexity = torch.exp(loss).item()
            perplexity_history.append(perplexity)
            batch_indices.append(batch_idx)
            
            wandb.log({
                "train_loss": loss.item(),
                "train_perplexity": perplexity,
                "batch": batch_idx
            })
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1

            # Log batch metrics
            if batch_idx % 1000 == 0:
                print(f"Batch {batch_idx}: Loss = {loss.item():.4f}, Perplexity = {perplexity:.4f}")
                wandb.log({
                    "batch_loss": loss.item(),
                    "batch_perplexity": perplexity,
                    "batch": batch_idx + epoch * num_batches
                })
        
        avg_loss = total_loss / num_batches
        avg_perplexity = np.exp(avg_loss)  # Average perplexity from average loss
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}, Perplexity = {avg_perplexity:.4f}")
        
        # Log epoch metrics
        wandb.log({
            "epoch": epoch + 1,
            "avg_loss": avg_loss,
            "avg_perplexity": avg_perplexity,
        })
    # Save the model state dict after training loop
        model_save_path = "trained_transformer_model_epoch_{epoch}.pt"
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")

# Generation
def infer():
    model_path = "trained_transformer_model.pt"
    
    # Create a new model instance with the same architecture
    inference_model = DecoderOnlyTransformer(
        vocab_size=processor.vocab_size,
        d_model=256, 
        num_layers=num_of_layers,
        num_heads=num_of_heads,
        d_ff=512,
        max_seq_len=context_length,
        dropout=0.1,
        fasttext_embeddings=embeddings,
        freeze_embeddings=False,
        pad_idx=processor.special_tokens['<pad>']
    )
    
    # Load the trained weights
    state_dict = torch.load(model_path, map_location="cpu")
    inference_model.load_state_dict(state_dict)
    inference_model.eval()
    inference_model = inference_model.to(device)
    print("Model loaded and moved to device")
    
    # Import visualization module
    from visualize_attention import visualize_attention_patterns
    
    evaluator = ModelEvaluator(inference_model, processor, device)
    print("Starting evaluation with first 5 tokens as prompt...")
    results = evaluator.evaluate_on_dataset(
        test_file="data/tinystories_val.txt",
        num_samples=50,
        max_length=50,
        prompt_length=5,  # Use first 5 tokens as prompt
        beam_search=False,
        beam_size=5
    )
    
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Number of samples: {results['num_samples']}")
    print(f"Average Perplexity: {results['avg_perplexity']:.4f}")
    print(f"BLEU Score: {results['bleu_score']:.4f}")
    print(f"BLEU-1: {results['bleu_scores']['precisions'][0]:.4f}")
    print(f"BLEU-2: {results['bleu_scores']['precisions'][1]:.4f}")
    print(f"BLEU-3: {results['bleu_scores']['precisions'][2]:.4f}")
    print(f"BLEU-4: {results['bleu_scores']['precisions'][3]:.4f}")
    
    print("\n" + "="*50)
    print("SAMPLE GENERATED TEXTS")
    print("="*50)
    for i, (gen, ref) in enumerate(zip(results['generated_texts'], results['references'])):
        print(f"\nSample {i+1}:")
        print(f"Generated: {gen}")
        print(f"Reference: {ref}")
    
    evaluator.save_results(results, "evaluation_results.json")
    print(f"\nEvaluation complete! Results saved to evaluation_results.json")
    
    # Generate attention visualizations for 2-3 example sentences
    print("\n" + "="*50)
    print("GENERATING ATTENTION VISUALIZATIONS")
    print("="*50)
    
    # Load a few example sentences from validation set
    with open("data/tinystories_val.txt", 'r', encoding='utf-8') as f:
        all_stories = [line.strip() for line in f if line.strip()]
    
    # Select 2-3 examples for visualization
    import random
    example_texts = random.sample(all_stories, min(3, len(all_stories)))
    
    for idx, example_text in enumerate(example_texts):
        print(f"\nVisualizing attention for example {idx+1}:")
        print(f"Text: {example_text[:100]}...")  # Print first 100 chars
        
        try:
            visualize_attention_patterns(
                model=inference_model,
                processor=processor,
                text=example_text,
                device=device,
                save_dir=f"attention_visualizations/example_{idx+1}"
            )
        except Exception as e:
            print(f"Error visualizing attention for example {idx+1}: {e}")
            import traceback
            traceback.print_exc()
    
    wandb.log({"generated_sample": results['generated_texts']})
    print("\nInference and visualization complete!")

import sys

def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py [train|infer|both]")
        sys.exit(1)
    action = sys.argv[1].lower()
    if action == 'train':
        train()
    elif action == 'infer':
        infer()
    elif action == 'both':
        train()
        infer()
    else:
        print("Invalid argument. Use 'train', 'infer', or 'both'.")
        sys.exit(1)

if __name__ == "__main__":
    main()

wandb.finish()

