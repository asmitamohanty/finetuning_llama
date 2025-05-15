import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from typing import Optional, List
from llama.model import Attention as LlamaAttention
from llama.model import ModelArgs, TransformerBlock, Llama
from torch.utils.checkpoint import checkpoint

class LoRALayer():
    def __init__(
        self, 
        r: int, 
        lora_alpha: int, 
        lora_dropout: float,
        merge_weights: bool,
    ):
        super(LoRALayer, self).__init__()
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        self.merged = False
        self.merge_weights = merge_weights

class Linear(nn.Linear, LoRALayer):
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        bias: bool = True, 
        r: int = 0,
        lora_alpha: int = 1, 
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False, # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        **kwargs
    ):
        self.fan_in_fan_out = fan_in_fan_out
        nn.Linear.__init__(self, in_features, out_features, bias=bias)
        LoRALayer.__init__(self, r, lora_alpha, lora_dropout,
                           merge_weights=merge_weights)
        if r > 0:
            self.lora_A = nn.Parameter(torch.zeros((r, in_features)), requires_grad=True).to(self.weight.device)
            self.lora_B = nn.Parameter(torch.zeros((out_features, r)), requires_grad=True).to(self.weight.device)
            self.scaling = self.lora_alpha / r
            
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)
    
    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True) -> 'LoRALayer':
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        
        super().train(mode)
        if mode: #training
            if self.merged and self.merge_weights: 
                # If we are in training mode and the weights are merged, unmerge them to update the lora A & B separately
                self.weight.data = T(self.weight.data) - self.scaling * (self.lora_B @ self.lora_A).t()
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # If we are in eval mode and the weights are not merged, merge them
                self.weight.data = T(self.weight.data) + self.scaling * (self.lora_B @ self.lora_A).t()
                self.merged = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        
        if self.r > 0 and not self.merged:
            # LoRA forward pass
            lora_out = self.lora_dropout(x) @ self.lora_A.t() @ self.lora_B.t()
            return F.linear(x, T(self.weight), bias=self.bias) + self.scaling * lora_out
        else: 
            # Standard linear forward pass, use when LoRA weights are merged
            return F.linear(x, T(self.weight), bias=self.bias)

class LoRAAttention(LlamaAttention):
      def __init__(self, args: ModelArgs):
          super().__init__(args)
          self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
          self.dim = args.dim
          self.head_dim = args.dim // args.n_heads
          self.wq = Linear(args.dim, args.n_heads * self.head_dim, r=16, 
                           lora_alpha=32, lora_dropout=0.05, bias=False)
          self.wv = Linear(args.dim, self.n_kv_heads * self.head_dim, r=16, 
                           lora_alpha=32, lora_dropout=0.05, bias=False)
 
class LoRATransformerBlock(TransformerBlock):
    def __init__(self, layer_id: int, args: ModelArgs, enable_ckpt: bool = False):
        super().__init__(layer_id, args)
        self.attention = LoRAAttention(args)
        self.enable_checkpointing = enable_ckpt

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        """
        Perform a forward pass through the TransformerBlock.

        Args:
            x (torch.Tensor): Input tensor.
            start_pos (int): Starting position for attention caching.
            freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.
            mask (torch.Tensor, optional): Masking tensor for attention. Defaults to None.

        Returns:
            torch.Tensor: Output tensor after applying attention and feedforward layers.

        """


        h = x + self.attention.forward(
            self.attention_norm(x), start_pos, freqs_cis, mask
        )

        normed = self.ffn_norm(h)
        if self.enable_checkpointing:
            out = h + checkpoint(self.feed_forward.forward, normed, use_reentrant=False)
        else:
            out = h + self.feed_forward.forward(normed) 
        return out

class LoRALlama(Llama):
    def __init__(self, args: ModelArgs, num_ckpt_layers: int = 0):
        super().__init__(args)
        self.layers = nn.ModuleList([LoRATransformerBlock(i, args, enable_ckpt=(i >= args.n_layers - num_ckpt_layers)) 
                                     for i in range(args.n_layers)])

def freeze_lora_layers(model):
    """
    Freeze the all model parameters except lora parameters.
    """
    for name, param in model.named_parameters():
        param.requires_grad = "lora_" in name