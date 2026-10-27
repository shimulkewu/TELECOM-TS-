#!/usr/bin/env python3
"""
Title: Mechanistic Interpretability Layer Interception Tool
Description: Implements PyTorch forward hooks to pull high-precision residual-stream 
             activation tensors from specific target layers of LLaMA-3-8B.
"""

import torch
import torch.nn as nn
from typing import Dict, List

# Define architectural constants matching LLaMA-3-8B parameter spaces
D_MODEL = 4096
LAYER_16 = 16
LAYER_24 = 24

class MockTransformerBlock(nn.Module):
    """Simulates the internal forward transformation logic of a single LLaMA layer."""
    def __init__(self, layer_id: int):
        super().__init__()
        self.layer_id = layer_id
        self.linear = nn.Linear(D_MODEL, D_MODEL)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.tanh(self.linear(x)) * 0.1

class TelecomLlamaMock(nn.Module):
    """Simulates a 32-layer LLaMA architecture stack for validation checks."""
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([MockTransformerBlock(i) for i in range(32)])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden_states = x
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states

def make_activation_hook(layer_name: str, storage_dict: Dict[str, torch.Tensor]):
    """Generates deterministic forward hooks to capture mid-pass activation matrices."""
    def hook(module: nn.Module, input_tensors: tuple, output_tensor: torch.Tensor):
        storage_dict[layer_name] = output_tensor.detach().cpu()
    return hook

def main():
    torch.manual_seed(42)
    schema_tokens = ["[SCH]", "META", "|", "TS", "5G"]
    mock_input_embeddings = torch.randn(len(schema_tokens), D_MODEL) * 0.5
    
    model = TelecomLlamaMock()
    intercepted_activations: Dict[str, torch.Tensor] = {}
    
    model.layers[LAYER_16].register_forward_hook(
        make_activation_hook(f"Layer_{LAYER_16}", intercepted_activations)
    )
    model.layers[LAYER_24].register_forward_hook(
        make_activation_hook(f"Layer_{LAYER_24}", intercepted_activations)
    )
  
    model.eval()
    with torch.no_grad():
        _ = model(mock_input_embeddings)
        
    print("="*24 + " RESIDUAL STREAM ACTIVATION EXTRACTION TRACE " + "="*23)
    print(f"Target Architecture: LLaMA-3-8B Fine-Tuned | Hidden Dimension (d_model): {D_MODEL}")
    print(f"Extraction Targets : Layer i = {LAYER_16}, Layer i = {LAYER_24}")
    print("="*72 + "\n")
    
    for layer_id in [LAYER_16, LAYER_24]:
        layer_key = f"Layer_{layer_id}"
        print(f"[LAYER_{layer_id}_ACTIVATION_STREAM]")
        
        tensor_data = intercepted_activations[layer_key]
        for token_idx, token in enumerate(schema_tokens):
            vector = tensor_data[token_idx]
            head_elements = ", ".join([f"{val.item():+.4f}" for val in vector[:4]])
            tail_element = f"{vector[-1].item():+.4f}"
            print(f" Token ({token:<6}) -> X_{layer_id} = [{head_elements}, ..., {tail_element}]^T")
            
        if layer_id == LAYER_16:
            print("\n" + "-"*72 + "\n")
            
    print("\n" + "="*72)

if __name__ == "__main__":
    main()
