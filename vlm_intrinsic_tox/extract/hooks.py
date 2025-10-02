"""Hooks for capturing residual streams and disabling cross attention."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, Iterator, List, Tuple

import torch

from ..utils.logging import get_logger


LOGGER = get_logger(__name__)


class ResidualCapture:
    """Callable hook to capture decoder layer residual streams."""

    def __init__(self) -> None:
        self.activations: Dict[int, torch.Tensor] = {}

    def __call__(self, layer_idx: int, hidden: torch.Tensor) -> None:
        self.activations[layer_idx] = hidden.detach().to("cpu")

    def clear(self) -> None:
        self.activations.clear()


@contextmanager
def cross_attention_off(layer) -> Iterator[None]:
    """Context manager that zeros the self-attention forward output."""

    original = layer.self_attn.forward

    def _disabled_forward(*args, **kwargs):
        hidden_states = args[0] if args else kwargs.get("hidden_states")
        if hidden_states is None:
            output = original(*args, **kwargs)
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
        zeros = torch.zeros_like(hidden_states)
        return (zeros, None)

    layer.self_attn.forward = _disabled_forward
    try:
        yield
    finally:
        layer.self_attn.forward = original


def install_post_attention_hook(model, layers: List[int], capture: ResidualCapture) -> List[torch.utils.hooks.RemovableHandle]:
    """Register forward hooks that capture the post-attention residual."""

    handles: List[torch.utils.hooks.RemovableHandle] = []
    decoder_layers = model.language_model.layers
    for idx in layers:
        layer = decoder_layers[idx]

        def _make_hook(i: int):
            def _hook(module, input_, output):
                hidden = output[0] if isinstance(output, tuple) else output
                capture(i, hidden)

            return _hook

        handles.append(layer.register_forward_hook(_make_hook(idx)))
    return handles


def remove_hooks(handles: List[torch.utils.hooks.RemovableHandle]) -> None:
    for handle in handles:
        handle.remove()


__all__ = ["ResidualCapture", "cross_attention_off", "install_post_attention_hook", "remove_hooks"]
