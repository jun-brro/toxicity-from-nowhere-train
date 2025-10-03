"""Concept-based steering mechanisms for VLM intervention."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import numpy as np

from ..utils.logging import get_logger

LOGGER = get_logger(__name__)


class GCAVSteering:
    """Implements GCAV (Generalized Concept Activation Vector) steering mechanism."""
    
    def __init__(self, cav_path: str | Path, target_layers: List[int], device: str = "cuda:0") -> None:
        """Initialize GCAV steering.
        
        Args:
            cav_path: Path to pre-trained CAV file
            target_layers: List of layer indices to apply steering
            device: Device to run computations on
        """
        self.device = device
        self.target_layers = target_layers
        self.cavs = self.load_cavs(cav_path)
        self.hooks = []
        self.intervention_stats = []  # Track interventions for analysis
        
    def load_cavs(self, cav_path: str | Path) -> Dict:
        """Load pre-trained CAVs from file.
        
        Args:
            cav_path: Path to CAV file (.pt or .pkl)
            
        Returns:
            Dictionary of CAVs indexed by layer
        """
        cav_path = Path(cav_path)
        
        if not cav_path.exists():
            raise FileNotFoundError(f"CAV file not found: {cav_path}")
        
        if cav_path.suffix == '.pt':
            cavs = torch.load(cav_path, map_location=self.device)
        else:
            with open(cav_path, 'rb') as f:
                cavs = pickle.load(f)
        
        # Convert numpy arrays to tensors if needed
        for layer in cavs:
            for key in ['w', 'b', 'v']:
                if key in cavs[layer] and not isinstance(cavs[layer][key], torch.Tensor):
                    cavs[layer][key] = torch.tensor(
                        cavs[layer][key], dtype=torch.float32, device=self.device
                    )
        
        LOGGER.info(f"Loaded CAVs for layers: {list(cavs.keys())}")
        return cavs
    
    def compute_concept_probability(self, activation: torch.Tensor, layer: int) -> torch.Tensor:
        """Compute concept probability P_d(e) = σ(w^T * e + b).
        
        Args:
            activation: Layer activation tensor
            layer: Layer index
            
        Returns:
            Concept probability tensor
        """
        if layer not in self.cavs:
            return torch.tensor(0.5, device=self.device, dtype=activation.dtype)

        cav = self.cavs[layer]
        # Ensure CAV tensors match activation dtype
        activation_dtype = activation.dtype
        w = cav['w'].to(device=self.device, dtype=activation_dtype)
        b = cav['b'].to(device=self.device, dtype=activation_dtype)
        
        # Apply same preprocessing as during training: mean across sequence dimension
        if activation.dim() > 2:
            # Shape: [batch, seq_len, hidden] -> [batch, hidden]
            pooled_activation = activation.mean(dim=1)
        else:
            pooled_activation = activation
        
        # Compute logits and probability
        logits = torch.matmul(pooled_activation, w) + b
        probability = torch.sigmoid(logits)
        
        return probability
    
    def compute_intervention_direction(self, activation: torch.Tensor, layer: int, 
                                     target_prob: float = 0.3, strength: float = 1.0) -> torch.Tensor:
        """Compute intervention direction using closed-form solution.
        
        Args:
            activation: Layer activation tensor
            layer: Layer index
            target_prob: Target concept probability
            strength: Intervention strength multiplier
            
        Returns:
            Intervention direction tensor
        """
        if layer not in self.cavs:
            return torch.zeros_like(activation)
        
        cav = self.cavs[layer]
        activation_dtype = activation.dtype
        w = cav['w'].to(device=self.device, dtype=activation_dtype)
        v = cav['v'].to(device=self.device, dtype=activation_dtype)
        
        # Pool activation if needed
        if activation.dim() > 2:
            pooled_activation = activation.mean(dim=1)
            original_shape = activation.shape
        else:
            pooled_activation = activation
            original_shape = None
        
        # Current concept probability
        current_prob = self.compute_concept_probability(activation, layer)
        
        # Compute intervention magnitude using closed-form solution
        # ε = (logit(p_target) - logit(p_current)) / ||w||²
        target_logit = torch.logit(torch.clamp(torch.tensor(target_prob), 1e-6, 1-1e-6))
        current_logit = torch.logit(torch.clamp(current_prob, 1e-6, 1-1e-6))
        
        w_norm_sq = torch.sum(w * w)
        epsilon = (target_logit - current_logit) / w_norm_sq
        
        # Apply strength multiplier
        epsilon = epsilon * strength
        
        # Compute intervention: Δe = ε * v
        intervention = epsilon.unsqueeze(-1) * v.unsqueeze(0)
        
        # Reshape back to original activation shape if needed
        if original_shape is not None:
            intervention = intervention.unsqueeze(1).expand(-1, original_shape[1], -1)
        
        # Store intervention stats
        self.intervention_stats.append({
            'layer': layer,
            'current_prob': current_prob.item(),
            'target_prob': target_prob,
            'epsilon': epsilon.item(),
            'intervention_norm': torch.norm(intervention).item()
        })
        
        return intervention
    
    def create_steering_hook(self, layer: int, target_prob: float = 0.3, 
                           strength: float = 1.0, direction: str = "suppress"):
        """Create a forward hook for steering interventions.
        
        Args:
            layer: Layer index to apply steering
            target_prob: Target concept probability
            strength: Intervention strength
            direction: "suppress" or "amplify"
            
        Returns:
            Forward hook function
        """
        def steering_hook(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            
            # Compute intervention
            intervention = self.compute_intervention_direction(
                hidden_states, layer, target_prob, strength
            )
            
            # Apply intervention based on direction
            if direction == "suppress":
                modified_states = hidden_states - intervention
            elif direction == "amplify":
                modified_states = hidden_states + intervention
            else:
                raise ValueError(f"Unknown direction: {direction}")
            
            # Return modified output
            if isinstance(output, tuple):
                return (modified_states,) + output[1:]
            else:
                return modified_states
        
        return steering_hook
    
    def install_hooks(self, model, config: Dict) -> None:
        """Install steering hooks on the model.
        
        Args:
            model: The model to install hooks on
            config: Configuration dictionary with steering parameters
        """
        self.remove_hooks()  # Remove any existing hooks
        
        for layer_idx in self.target_layers:
            if layer_idx >= len(model.language_model.layers):
                LOGGER.warning(f"Layer {layer_idx} not found in model")
                continue
            
            layer_module = model.language_model.layers[layer_idx]
            
            # Get steering config for this layer
            layer_config = config.get(f"layer_{layer_idx}", config.get("default", {}))
            
            hook = self.create_steering_hook(
                layer=layer_idx,
                target_prob=layer_config.get("target_prob", 0.3),
                strength=layer_config.get("strength", 1.0),
                direction=layer_config.get("direction", "suppress")
            )
            
            handle = layer_module.register_forward_hook(hook)
            self.hooks.append(handle)
            
            LOGGER.info(f"Installed steering hook on layer {layer_idx}")
    
    def remove_hooks(self) -> None:
        """Remove all installed hooks."""
        for handle in self.hooks:
            handle.remove()
        self.hooks.clear()
        LOGGER.info("Removed all steering hooks")
    
    def get_intervention_stats(self) -> List[Dict]:
        """Get statistics about applied interventions.
        
        Returns:
            List of intervention statistics
        """
        return self.intervention_stats.copy()
    
    def clear_stats(self) -> None:
        """Clear intervention statistics."""
        self.intervention_stats.clear()


__all__ = ["GCAVSteering"]