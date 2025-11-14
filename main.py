import torch
import torch.nn.functional as F
import wandb
from tokenizer import prepare_data_pipeline
from model_architecture import DecoderOnlyTransformer

wandb.init(project="transformer-training", name="decoder-only")

processor, X_batch, y_batch, embeddings = prepare_data_pipeline(
    stories_file="data/tinystories.txt",
    cache_dir="cache",
    batch_size=32,
    use_fasttext=True,
    fasttext_path="cc.en.300.vec"
)
print(f"Vocab size: {processor.vocab_size}")

model = DecoderOnlyTransformer(
    vocab_size=processor.vocab_size,
    d_model=256, 
    num_layers=1,
    num_heads=8,
    d_ff=512,
    max_seq_len=128,
    dropout=0.1,
    fasttext_embeddings=embeddings,
    freeze_embeddings=False,  # Allow fine-tuning
    pad_idx=processor.special_tokens['<pad>']
)

num_params = model.get_num_parameters()
print(f"Model parameters: {num_params:,}")
wandb.log({"num_parameters": num_params})

# Training loop
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
print(f"Model moved to device: {device}")
optimizer = model.configure_optimizers(learning_rate=3e-4)
print(f"Optimizer created")

model.train()
for epoch in range(1):
    # Get batches
    data_generator = processor.prepare_training_data("data/tinystories.txt", batch_size=32)
    print(f"Data generator created")
    total_loss = 0
    num_batches = 0
    
    for batch_idx, (X_batch, y_batch) in enumerate(data_generator):
        print(f"Batch {batch_idx} created")
        # Move to device
        X_batch = torch.tensor(X_batch, dtype=torch.long).to(device)
        y_batch = torch.tensor(y_batch, dtype=torch.long).to(device)
        logits = model(X_batch)
        
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y_batch.reshape(-1),
            ignore_index=processor.special_tokens["<pad>"]
        )
        wandb.log({"train_loss": loss.item()})
        # print(f"Loss calculated, loss: {loss.item()}")
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1

        # Log batch metrics
        if batch_idx % 1 == 0:
            wandb.log({
                "batch_loss": loss.item(),
                "batch": batch_idx + epoch * num_batches
            })
    
    avg_loss = total_loss / num_batches
    print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")
    
    # Log epoch metrics
    wandb.log({
        "epoch": epoch + 1,
        "avg_loss": avg_loss,
    })

# Generation
model.eval()
print(f"Model evaluated")
start_tokens = torch.tensor([[processor.special_tokens['<sos>']]], device=device)
generated = model.generate(
    start_tokens, 
    max_length=50, 
    temperature=0.8,
    top_k=50,
    eos_token_id=processor.special_tokens['<eos>']
)
generated_text = processor.ids_to_text(generated[0].cpu().tolist())
print(f"\nGenerated: {generated_text}")
wandb.log({"generated_sample": generated_text})

wandb.finish()