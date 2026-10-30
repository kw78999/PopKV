"""Shared decode execution helpers; independent of the attention method."""
from __future__ import annotations

import torch


class DecodeEventPool:
    """Create and initialize CUDA events before measuring any decode token.

    One decoder owns one pool and executes tokens sequentially. finish() waits
    for the final event before a subsequent token reuses these handles.
    """
    def __init__(self, device: torch.device, capacity: int) -> None:
        self.device = device
        self.events = []
        with torch.cuda.device(device):
            stream = torch.cuda.current_stream(device)
            for _ in range(capacity):
                event = torch.cuda.Event(enable_timing=True)
                event.record(stream)  # Force CUDA's otherwise lazy initialization.
                self.events.append(event)
            stream.synchronize()
        self.cursor = 0

    def reset(self) -> None:
        self.cursor = 0

    def record(self):
        if self.cursor >= len(self.events):
            raise RuntimeError("decode event pool exhausted; increase reserved capacity")
        event = self.events[self.cursor]
        self.cursor += 1
        event.record(torch.cuda.current_stream(self.device))
        return event


def mlp_block(layer, hidden: torch.Tensor) -> torch.Tensor:
    """Historical MLP timing boundary: post-attention norm + MLP + residual."""
    return hidden + layer.mlp(layer.post_attention_layernorm(hidden))


class CUDAGraphMLP:
    """Capture one layer's fixed-shape one-token MLP, with explicit input copy.

    Setup is outside decode timing. Output storage is reused on each replay;
    callers must consume it before replaying this layer. No model weights are
    copied. Rebuild after changing weights, device, dtype, hooks or model mode.
    """
    @torch.inference_mode()
    def __init__(self, layer, example: torch.Tensor) -> None:
        if example.device.type != "cuda":
            raise ValueError("CUDA Graph MLP requires a CUDA device")
        if layer.training:
            raise ValueError("CUDA Graph MLP requires model.eval()")
        self.signature = (tuple(example.shape), example.dtype, example.device)
        self.static_input = torch.empty_like(example).copy_(example)
        with torch.cuda.device(example.device):
            current = torch.cuda.current_stream(example.device)
            warmup = torch.cuda.Stream(device=example.device)
            warmup.wait_stream(current)
            with torch.cuda.stream(warmup):
                for _ in range(3):
                    output = mlp_block(layer, self.static_input)
            current.wait_stream(warmup)
            warmup.synchronize()
            del output
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.static_output = mlp_block(layer, self.static_input)

    @torch.inference_mode()
    def __call__(self, hidden: torch.Tensor) -> torch.Tensor:
        if (tuple(hidden.shape), hidden.dtype, hidden.device) != self.signature:
            raise ValueError("MLP replay shape/dtype/device changed; rebuild graphs")
        self.static_input.copy_(hidden)
        self.graph.replay()
        return self.static_output


def method_order(methods, round_index: int):
    """Rotate and reverse a complete method set; every2n rounds balances positions."""
    methods = tuple(methods)
    if not methods:
        raise ValueError("at least one method is required")
    n = len(methods)
    base = methods if (round_index // n) % 2 == 0 else tuple(reversed(methods))
    offset = round_index % n
    return base[offset:] + base[:offset]
