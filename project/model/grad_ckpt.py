import torch
import torch.nn as nn
import torch.nn.functional as F
from llama.model import Attention as LlamaAttention
from llama.model import ModelArgs, TransformerBlock, Llama
from torch.utils.checkpoint import checkpoint
from typing import Optional, List

class GCTransformerBlock(TransformerBlock):
    def __init__(self, layer_id: int, args: ModelArgs, enable_ckpt: bool = False):
        super().__init__(layer_id, args)
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

class GCLlama(Llama):
    def __init__(self, args: ModelArgs, num_ckpt_layers: int = 0):
        super().__init__(args)
        self.layers = nn.ModuleList([GCTransformerBlock(i, args, enable_ckpt=(i >= args.n_layers - num_ckpt_layers)) 
                                     for i in range(args.n_layers)])