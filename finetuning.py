import torch
from torch.optim import SGD
from torch.utils.data import DataLoader

from llama.model import ModelArgs, Llama
from alpaca_dataset import AlpacaDataModule, IGNORE_INDEX
import argparse
from project.model.lora import LoRALlama, freeze_lora_layers
from project.model.grad_ckpt import GCLlama
from torch.cuda.amp import GradScaler, autocast
import sys
from project.utils import register_hooks, WandbLogger, print_model_parameters
from tqdm import tqdm
import time

SAMPLE_SIZE = 200
EPOCHS = 10

def train(
    tokenizer_model_path: str,
    data_path: str,
    checkpoint_path: str,
    use_lora: str,
    use_grad_ckpt: str,
    use_grad_acc: str,
    use_mixed_prec: str,
    batch_size: int = 1,
    lr: float = 1e-5,
    num_epochs: int = EPOCHS,
    accumulation_steps: int = 1,
):
    model_type = ""
    if use_grad_ckpt == "1" and use_lora == "1" and use_mixed_prec == "1":
        model_type = "LoRA + Gradient Checkpointing + Mixed Precision"
    elif use_lora == "1" and use_grad_ckpt == "1":
        model_type = "LoRA + Gradient Checkpointing"
    elif use_lora == "1" and use_mixed_prec == "1":
        model_type = "LoRA + Mixed Precision"
    elif use_grad_ckpt == "1" and use_mixed_prec == "1":
        model_type = "Gradient Checkpointing + Mixed Precision"
    elif use_lora == "1":
        model_type = "LoRA"
    elif use_grad_ckpt == "1":
        model_type = "Gradient Checkpointing"
    elif use_grad_acc == "1":
        model_type = "Gradient Accumulation"   
    elif use_mixed_prec == "1":
        model_type = "Mixed Precision"
    else:
        model_type = "Vanilla Llama"
    print(f"Model Finetuning: {model_type}")

    print("Calling AlpacaDataModule...")
    module = AlpacaDataModule(tokenizer_model_path, data_path, sample_size=SAMPLE_SIZE)
    module.tok.pad_id = module.tok.eos_id
    train_ds = module.train_dataset
    collator = module.data_collator
    print("Length of train dataset:", len(train_ds))

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator
    )

    print("Length of train loader:", len(train_loader))
    # Load model architecture and pretrained weights
    print("Loading model...")
    args = ModelArgs()
    args.kv_caching = False
    num_ckpt_layers = 0
    if use_grad_ckpt == "1":
        num_ckpt_layers = 4
        print(f"Checkpointing enabled for last {num_ckpt_layers} layers")
    if use_lora == "1":
        model = LoRALlama(args, num_ckpt_layers = num_ckpt_layers)
    elif use_grad_ckpt == "1" and not use_lora == "1" and not use_mixed_prec == "1" and not use_grad_acc == "1":
        model = GCLlama(args, num_ckpt_layers = num_ckpt_layers)
    else:
        model = Llama(args)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint, strict=False)
    
    print("Mapping model to device...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    if hasattr(model, 'freqs_cis'):
        model.freqs_cis = model.freqs_cis.to(device)

    model.train()
    if use_lora == "1":
        freeze_lora_layers(model)
    
    print_model_parameters(model, model_type)
    
    optimizer = SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, momentum=0)
    scaler = GradScaler() 

    #Setup Wandb
    print("Setup Wandb...")
    logger = WandbLogger(model_type)
    activation_cache = []
    hooks = register_hooks(model, activation_cache)

    # Training loop
    print("Beginning training...")
    
    if use_grad_acc == "1":
        accumulation_steps = 8
    train_losses = []
    global_peak_mem = 0
    total_start_time = time.time()
    for epoch in range(num_epochs):
        train_loss = 0.0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc="Training")
        torch.cuda.reset_peak_memory_stats(device)
        step_times = []
        epoch_activation_mem = 0
        
        for idx, batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            labels    = batch["labels"].to(device)
            batch_start_time = time.time()
            
            activation_cache.clear()
            optimizer.zero_grad()
            if use_mixed_prec == "1":
                with autocast():
                    logits = model(input_ids, start_pos=0)
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()

                    loss = torch.nn.functional.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                        ignore_index=IGNORE_INDEX
                    )
                scaler.scale(loss).backward()
            
            else: 
            
                logits = model(input_ids, start_pos=0)
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()

                loss = torch.nn.functional.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=IGNORE_INDEX
                )
            
                loss.backward()
        
            if (idx + 1) % accumulation_steps == 0:
                if use_mixed_prec == "1":
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
            
            sys.stdout.write(f"\r Processing... {idx*100/len(train_loader)} %")  
            sys.stdout.flush()  

            train_loss += loss.item()
            step_time = time.time() - batch_start_time
            step_times.append(step_time)
            batch_mem = logger.estimate_activation_mem(activation_cache)
            epoch_activation_mem += batch_mem
   
            progress_bar.set_postfix({"loss": f"{loss.item():.2f}"})
            batch_idx_step = epoch * len(train_loader) + idx
            #print(f"batch_idx_step: {batch_idx_step} loss: {loss.item():.4f} step time: {step_time:.4f}")

            logger.log_loss_per_step(batch_idx_step, model_type, loss.item())

        avg_epoch_loss = train_loss / len(train_loader)
        #Epoch-level memory stats
        logger.log_system_metrics(epoch, model, optimizer, epoch_activation_mem)
        logger.log_loss_epoch(epoch, model_type, avg_epoch_loss)
        train_losses.append(avg_epoch_loss)
        peak_mem_bytes = torch.cuda.max_memory_allocated()
        peak_mem_mb = peak_mem_bytes / (1024 ** 2)
        global_peak_mem = max(global_peak_mem, peak_mem_mb)
        avg_step_time_ = sum(step_times) / len(step_times)
        
        print(f"Epoch {epoch+1} Avg loss (train): {avg_epoch_loss:.4f} Peak Mem/epochL {peak_mem_mb} MB avg step time {avg_step_time_:.4f} s")

        total_training_time = time.time() - total_start_time
    peak_memory = global_peak_mem
    print(f"Training with {model_type} completed in {total_training_time:.2f} s with peak memory at {peak_memory} MB.")
    
    # --- Final logging ---
    custom_step = num_epochs + 1
    logger.summary_static_metric(model_type, custom_step, "Global_Peak_Mem_MB", global_peak_mem)
    logger.summary_static_metric(model_type, custom_step, "Total_Training_Time_Sec", total_training_time)

    output_path = f"{model_type}_finetuned_llama3.2-1B.pth"
    torch.save(model.state_dict(), output_path)
    print(f"Fine-tuned model saved to {output_path}")

    logger.finish()
    for h in hooks:
        h.remove()
    
if __name__ == "__main__":
    print("Initializing...")
    parser = argparse.ArgumentParser(
        description="Train or run a LLaMA model with optional Fine Tuning Technique."
    )
    parser.add_argument("--tokenizer_model_path", type=str, required=True, 
                        help="Path to the tokenizer model. Eg: /<local_path>/.llama/checkpoints/Llama3.2-1B/tokenizer.model")
    parser.add_argument("--checkpoint_path", type=str, required=True, 
                        help="Path to the model checkpoint. Eg: /<local_path>/.llama/checkpoints/Llama3.2-1B/consolidated.00.pth")
    parser.add_argument("--use_lora", type=str, default=0, help="Enable (1) or Disable (0) LoRA")
    parser.add_argument("--use_grad_acc", type=str, default=0, help="Enable (1) or Disable (0) Gradient Accumulation")
    parser.add_argument("--use_grad_ckpt", type=str, default=0, help="Enable (1) or Disable (0) Gradient Checkpointing")
    parser.add_argument("--use_mixed_prec", type=str, default=0, help="Enable (1) or Disable (0) Mixed Precision")
    args = parser.parse_args()
    train(
        tokenizer_model_path=args.tokenizer_model_path,
        data_path="alpaca_data.json",
        checkpoint_path= args.checkpoint_path,
        use_lora=args.use_lora,
        use_grad_ckpt=args.use_grad_ckpt,
        use_grad_acc=args.use_grad_acc,
        use_mixed_prec=args.use_mixed_prec,
    )
