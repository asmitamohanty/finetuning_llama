import wandb
import torch

class WandbLogger:
    def __init__(self, model_name):
        wandb.login(relogin=True)
        wandb.init(project="FineTuning Perf Metrics", name=model_name)
        print(f"WandB logging initialized for project: Llama3.2-1B FineTuning, run: {model_name}")
        wandb.define_metric("batch_step")
        wandb.define_metric("epoch_step")
        wandb.define_metric(f"{model_name} Avg Loss/epoch", step_metric="epoch_step")
        wandb.define_metric(f"{model_name} Loss/step", step_metric="batch_step")
        wandb.define_metric(f"Per Epoch/{model_name}_Peak_Memory_MB", step_metric="epoch_step")
        wandb.define_metric(f"Per Epoch/{model_name}_RunTime_Train_Sec", step_metric="epoch_step")
        wandb.define_metric("Memory per epoch/Trainable_Params_MB", step_metric="epoch_step")
        wandb.define_metric("Memory per epoch/Activations_MB", step_metric="epoch_step")
        wandb.define_metric("Memory per epoch/Gradients_MB", step_metric="epoch_step")
        wandb.define_metric("Memory per epoch/Optimizer_State_MB", step_metric="epoch_step")
        wandb.define_metric(f"{model_name}_Global_Peak_Memory_MB", step_metric="epoch_step")
        wandb.define_metric(f"{model_name}_Global_RunTime_Train_Sec", step_metric="epoch_step")

    def finish(self):
        wandb.finish()
        print("WandB logging finished.")

    def estimate_param_mem(self, model):
        return sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad) / (1024 ** 2)

    def estimate_grad_mem(self, model):
        return sum(p.grad.numel() * p.grad.element_size() for p in model.parameters()
                if p.grad is not None and p.requires_grad) / (1024 ** 2)

    def estimate_optimizer_mem(self, optimizer):
        total = 0
        for group in optimizer.state.values():
            for state_val in group.values():
                if torch.is_tensor(state_val):
                    total += state_val.numel() * state_val.element_size()
        return total / (1024 ** 2)

    def estimate_activation_mem(self, activation_cache):
        return sum(t.numel() * t.element_size() for t in activation_cache) / (1024 ** 2)

    def log_system_metrics(self, epoch, model, optimizer, activation_mem):
        param_mem = self.estimate_param_mem(model)
        grad_mem = self.estimate_grad_mem(model)
        opt_mem = self.estimate_optimizer_mem(optimizer)

        # Log to WandB
        wandb.log({
            "Memory per epoch/Trainable_Params_MB": param_mem,
            "Memory per epoch/Activations_MB": activation_mem,
            "Memory per epoch/Gradients_MB": grad_mem,
            "Memory per epoch/Optimizer_State_MB": opt_mem,
            "epoch_step": epoch+1,
        })#, step=epoch+1)

    def log_final_metrics(self, model_name, step, peak_mem, runtime):

        wandb.log({
            f"Per Epoch/{model_name}_Peak_Memory_MB": peak_mem,
            f"Per Epoch/{model_name}_RunTime_Train_Sec": runtime,
            "epoch_step": step,
        })#, step=step)
    
    def log_loss_per_step(self, steps, model_name, loss):
        
        wandb.log({
            f"{model_name} Loss/step": loss,
            "batch_step": steps,
        })#, step=steps)

    def log_loss_epoch(self, steps, model_name, loss):
        
        wandb.log({
            f"{model_name} Avg Loss/epoch": loss,
            "epoch_step": steps,
        })#, step=steps)
        

    def summary_static_metric(self, model_name, custom_step, metric_name, metric_value):
        wandb.summary[f"{metric_name}"] = metric_value
        self.log_summary_metrics(model_name, custom_step, [metric_name], [metric_value])
    
    def log_summary_metrics(self, model_name, custom_step, metric_labels, metric_values):

        data = [[label, val] for label, val in zip(metric_labels, metric_values)]
        table = wandb.Table(data=data, columns=["Metric", "Value"])
        print(f"Logging summary metrics: {metric_labels} with values: {metric_values}")

        wandb.log({f"{model_name}_{metric_labels[0]}": 
                   wandb.plot.bar(table, "Metric", "Value", title="Metric Performance"), 
                   "epoch_step": custom_step})#, step=custom_step)

def print_model_parameters(model, finetuning_type):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{finetuning_type} Total parameters: {total_params}, Trainable parameters: {trainable_params}")
    print(f"Percentage of trainable parameters: {100*(trainable_params / total_params):.2f}%")

def get_activation_hook(activation_cache):
    def hook(module, input, output):
        outputs = output if isinstance(output, (list, tuple)) else [output]
        for o in outputs:
            if isinstance(o, torch.Tensor) and o.requires_grad:
                activation_cache.append(o.detach())
    return hook

def register_hooks(model, activation_cache):
    hooks = []
    activation_hook_fn = get_activation_hook(activation_cache)
    for _, module in model.named_modules():
        hooks.append(module.register_forward_hook(activation_hook_fn))
    return hooks