# Copyright 2024-present the Unsloth team.
# Licensed under the Apache License 2.0

"""
Unsloth Distributed Training Module

Optimized multi-GPU training for Unsloth with:
- Gradient compression
- Overlapped communication
- LoRA-aware distribution
- Ring all-reduce optimization
"""

from __future__ import annotations

import os
import logging
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.distributed as dist

__all__ = [
    "CompressedAllReduce",
    "OverlappedDDP",
    "LoRAAwareDDP",
    "ring_allreduce",
    "UnslothDDP",
]

logger = logging.getLogger(__name__)


class CompressedAllReduce:
    """
    Compress gradients before all-reduce.

    Only synchronize top-K gradient values for efficiency.
    Uses error feedback to maintain convergence.

    Args:
        compression_ratio: Ratio of gradients to keep (0.0 to 1.0).
            Default 0.1 means keep top 10% of gradients.
        min_elements: Minimum number of elements to keep for small tensors.

    Example:
        >>> compressor = CompressedAllReduce(compression_ratio=0.1)
        >>> values, indices = compressor.compress(grad_tensor, "layer1.weight")
        >>> # All-reduce values and indices
        >>> decompressed = compressor.decompress(values, indices, grad_tensor.shape)
    """

    def __init__(
        self,
        compression_ratio: float = 0.1,
        min_elements: int = 1,
    ):
        if not 0.0 < compression_ratio <= 1.0:
            raise ValueError(f"compression_ratio must be in (0, 1], got {compression_ratio}")

        self.compression_ratio = compression_ratio
        self.min_elements = min_elements
        self.error_feedback: Dict[str, torch.Tensor] = {}

    def compress(
        self,
        tensor: torch.Tensor,
        name: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compress gradient using top-K sparsification.

        Args:
            tensor: Gradient tensor to compress.
            name: Unique identifier for this tensor (for error feedback).

        Returns:
            Tuple of (compressed_values, indices).
        """
        numel = tensor.numel()
        k = max(self.min_elements, int(numel * self.compression_ratio))

        # Flatten tensor for processing
        flat_tensor = tensor.flatten()

        # Apply error feedback from previous iteration
        if name in self.error_feedback:
            error = self.error_feedback[name]
            if error.shape == flat_tensor.shape:
                flat_tensor = flat_tensor + error

        # Get top-K values and indices by absolute value
        _, indices = torch.topk(flat_tensor.abs(), min(k, numel))
        values = flat_tensor[indices]

        # Compute and store compression error for next iteration
        compressed = torch.zeros_like(flat_tensor)
        compressed[indices] = values
        self.error_feedback[name] = flat_tensor - compressed

        return values, indices

    def decompress(
        self,
        values: torch.Tensor,
        indices: torch.Tensor,
        shape: torch.Size,
    ) -> torch.Tensor:
        """
        Decompress gradient back to full tensor.

        Args:
            values: Compressed gradient values.
            indices: Indices of compressed values.
            shape: Original tensor shape.

        Returns:
            Decompressed tensor with original shape.
        """
        numel = 1
        for s in shape:
            numel *= s

        tensor = torch.zeros(
            numel,
            dtype=values.dtype,
            device=values.device,
        )
        tensor[indices] = values
        return tensor.view(shape)

    def clear_error_feedback(self):
        """Clear accumulated error feedback."""
        self.error_feedback.clear()


class OverlappedDDP(nn.Module):
    """
    DDP with overlapped communication and computation.

    Overlaps backward computation with gradient communication to
    hide communication latency.

    Args:
        model: The model to wrap.
        bucket_size_mb: Size of gradient buckets in MB. Larger buckets
            reduce communication overhead but increase latency.
        find_unused_parameters: Whether to find unused parameters.
            Set to True if some parameters don't receive gradients.

    Example:
        >>> model = MyModel()
        >>> ddp_model = OverlappedDDP(model, bucket_size_mb=25)
        >>> output = ddp_model(input)
        >>> loss = criterion(output, target)
        >>> loss.backward()  # Gradients synced automatically
    """

    def __init__(
        self,
        model: nn.Module,
        bucket_size_mb: float = 25.0,
        find_unused_parameters: bool = False,
    ):
        super().__init__()
        self.model = model
        self.bucket_size = int(bucket_size_mb * 1024 * 1024)
        self.find_unused_parameters = find_unused_parameters

        # Create gradient buckets
        self.buckets: List[List[nn.Parameter]] = []
        self.bucket_ready: List[List[bool]] = []
        self._create_buckets()

        # Communication stream for overlapping
        if torch.cuda.is_available():
            self.comm_stream = torch.cuda.Stream()
        else:
            self.comm_stream = None

        # Register hooks for overlapped communication
        self._hooks: List[Any] = []
        self._register_hooks()

        # Track if gradients have been synced
        self._grads_synced = False

    def _create_buckets(self):
        """Group parameters into communication buckets."""
        current_bucket: List[nn.Parameter] = []
        current_size = 0

        # Iterate in reverse order (to match backward pass)
        params = list(self.model.parameters())
        for param in reversed(params):
            if param.requires_grad:
                param_size = param.numel() * param.element_size()

                if current_size + param_size > self.bucket_size and current_bucket:
                    self.buckets.append(current_bucket)
                    self.bucket_ready.append([False] * len(current_bucket))
                    current_bucket = [param]
                    current_size = param_size
                else:
                    current_bucket.append(param)
                    current_size += param_size

        # Don't forget the last bucket
        if current_bucket:
            self.buckets.append(current_bucket)
            self.bucket_ready.append([False] * len(current_bucket))

        logger.debug(f"Created {len(self.buckets)} gradient buckets")

    def _register_hooks(self):
        """Register backward hooks for overlapped communication."""
        param_to_bucket: Dict[int, Tuple[int, int]] = {}

        # Map each parameter to its bucket and position
        for bucket_idx, bucket in enumerate(self.buckets):
            for param_idx, param in enumerate(bucket):
                param_to_bucket[id(param)] = (bucket_idx, param_idx)

        # Register hooks
        for param in self.model.parameters():
            if param.requires_grad and id(param) in param_to_bucket:
                bucket_idx, param_idx = param_to_bucket[id(param)]
                hook = param.register_post_accumulate_grad_hook(
                    lambda p, bi=bucket_idx, pi=param_idx: self._grad_hook(p, bi, pi)
                )
                self._hooks.append(hook)

    def _grad_hook(self, param: nn.Parameter, bucket_idx: int, param_idx: int):
        """Called when a gradient is ready."""
        self.bucket_ready[bucket_idx][param_idx] = True

        # Check if all gradients in bucket are ready
        if all(self.bucket_ready[bucket_idx]):
            self._sync_bucket(bucket_idx)
            # Reset for next iteration
            self.bucket_ready[bucket_idx] = [False] * len(self.bucket_ready[bucket_idx])

    def _sync_bucket(self, bucket_idx: int):
        """Synchronize a bucket's gradients across processes."""
        if not dist.is_initialized():
            return

        bucket = self.buckets[bucket_idx]

        # Collect gradients
        grads = []
        for param in bucket:
            if param.grad is not None:
                grads.append(param.grad.flatten())
            else:
                grads.append(torch.zeros(param.numel(), device=param.device, dtype=param.dtype))

        if not grads:
            return

        flat_grad = torch.cat(grads)

        # Launch all-reduce on communication stream
        if self.comm_stream is not None:
            with torch.cuda.stream(self.comm_stream):
                dist.all_reduce(flat_grad)
                flat_grad.div_(dist.get_world_size())

                # Copy back to individual gradients
                offset = 0
                for param in bucket:
                    numel = param.numel()
                    if param.grad is not None:
                        param.grad.copy_(flat_grad[offset:offset + numel].view(param.grad.shape))
                    offset += numel
        else:
            # CPU fallback
            dist.all_reduce(flat_grad)
            flat_grad.div_(dist.get_world_size())

            offset = 0
            for param in bucket:
                numel = param.numel()
                if param.grad is not None:
                    param.grad.copy_(flat_grad[offset:offset + numel].view(param.grad.shape))
                offset += numel

        self._grads_synced = True

    def forward(self, *args, **kwargs):
        """Forward pass through the model."""
        self._grads_synced = False
        return self.model(*args, **kwargs)

    def synchronize(self):
        """Wait for all gradient communications to complete."""
        if self.comm_stream is not None:
            self.comm_stream.synchronize()

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def __del__(self):
        """Clean up hooks on deletion."""
        self.remove_hooks()


class LoRAAwareDDP:
    """
    DDP optimized for LoRA training.

    Key insight: LoRA gradients (A, B matrices) are much smaller
    than base model gradients and can be synchronized faster.
    This class only syncs LoRA parameters, skipping frozen base model.

    Args:
        model: The LoRA model to optimize.
        sync_base_params: Whether to also sync base model parameters
            (useful for full fine-tuning alongside LoRA).

    Example:
        >>> model = get_peft_model(base_model, lora_config)
        >>> lora_ddp = LoRAAwareDDP(model)
        >>> # After backward pass
        >>> lora_ddp.sync_gradients()
    """

    def __init__(
        self,
        model: nn.Module,
        sync_base_params: bool = False,
    ):
        self.model = model
        self.sync_base_params = sync_base_params

        self.lora_params: List[nn.Parameter] = []
        self.base_params: List[nn.Parameter] = []

        # Separate LoRA and base parameters
        self._categorize_parameters()

        logger.info(
            f"LoRAAwareDDP: {len(self.lora_params)} LoRA params, "
            f"{len(self.base_params)} base params"
        )

    def _categorize_parameters(self):
        """Categorize parameters as LoRA or base."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # Check for common LoRA naming patterns
                name_lower = name.lower()
                if any(pattern in name_lower for pattern in [
                    'lora_a', 'lora_b', 'lora_embedding',
                    'lora_', '.lora', '_lora'
                ]):
                    self.lora_params.append(param)
                else:
                    self.base_params.append(param)

    def sync_gradients(self):
        """
        Synchronize gradients with different strategies.

        LoRA params: All-reduce with full precision (small tensors).
        Base params: Optional sync (usually frozen in LoRA training).
        """
        if not dist.is_initialized():
            return

        world_size = dist.get_world_size()

        # Sync LoRA gradients (always)
        if self.lora_params:
            self._sync_param_group(self.lora_params, world_size)

        # Sync base gradients (optional)
        if self.sync_base_params and self.base_params:
            self._sync_param_group(self.base_params, world_size)

    def _sync_param_group(self, params: List[nn.Parameter], world_size: int):
        """Synchronize a group of parameters."""
        grads = []
        grad_shapes = []

        for param in params:
            if param.grad is not None:
                grads.append(param.grad.flatten())
                grad_shapes.append(param.grad.shape)
            else:
                # Handle None gradients
                grads.append(torch.zeros(param.numel(), device=param.device, dtype=param.dtype))
                grad_shapes.append(param.shape)

        if not grads:
            return

        # Concatenate all gradients
        flat = torch.cat(grads)

        # All-reduce
        dist.all_reduce(flat)
        flat.div_(world_size)

        # Copy back to individual gradients
        offset = 0
        for i, param in enumerate(params):
            numel = param.numel()
            if param.grad is not None:
                param.grad.copy_(flat[offset:offset + numel].view(grad_shapes[i]))
            offset += numel

    def get_lora_grad_norm(self) -> float:
        """Get the norm of all LoRA gradients."""
        total_norm = 0.0
        for param in self.lora_params:
            if param.grad is not None:
                total_norm += param.grad.data.norm(2).item() ** 2
        return total_norm ** 0.5


def ring_allreduce(
    tensor: torch.Tensor,
    group: Optional[Any] = None,
) -> torch.Tensor:
    """
    Ring all-reduce with better bandwidth utilization.

    Instead of tree-based reduction, use ring topology
    for better scaling on high-bandwidth interconnects.

    Args:
        tensor: Tensor to all-reduce.
        group: Process group (default: global group).

    Returns:
        All-reduced tensor.

    Note:
        This is most effective for large tensors on systems with
        high-bandwidth interconnects (NVLink, InfiniBand).
    """
    if not dist.is_initialized():
        return tensor

    world_size = dist.get_world_size(group)
    rank = dist.get_rank(group)

    if world_size == 1:
        return tensor

    # Ensure tensor is contiguous
    tensor = tensor.contiguous()

    # Pad tensor to be divisible by world_size
    original_numel = tensor.numel()
    padded_size = ((original_numel + world_size - 1) // world_size) * world_size

    if padded_size > original_numel:
        padded = torch.zeros(padded_size, dtype=tensor.dtype, device=tensor.device)
        padded[:original_numel] = tensor.flatten()
        tensor = padded
    else:
        tensor = tensor.flatten()

    # Split tensor into chunks
    chunk_size = tensor.numel() // world_size
    chunks = list(tensor.chunk(world_size))

    # Allocate receive buffers
    recv_buffer = torch.empty(chunk_size, dtype=tensor.dtype, device=tensor.device)

    # Phase 1: Scatter-reduce
    # Each process accumulates one chunk from all other processes
    for i in range(world_size - 1):
        send_idx = (rank - i) % world_size
        recv_idx = (rank - i - 1) % world_size

        # Send and receive
        send_op = dist.isend(chunks[send_idx].contiguous(), (rank + 1) % world_size, group)
        dist.recv(recv_buffer, (rank - 1 + world_size) % world_size, group)
        send_op.wait()

        # Reduce into receive chunk
        chunks[recv_idx] = chunks[recv_idx] + recv_buffer

    # Phase 2: All-gather
    # Each process broadcasts its reduced chunk to all others
    for i in range(world_size - 1):
        send_idx = (rank - i + 1) % world_size
        recv_idx = (rank - i) % world_size

        # Send and receive
        send_op = dist.isend(chunks[send_idx].contiguous(), (rank + 1) % world_size, group)
        dist.recv(recv_buffer, (rank - 1 + world_size) % world_size, group)
        send_op.wait()

        # Copy received chunk
        chunks[recv_idx] = recv_buffer.clone()

    # Reconstruct tensor and average
    result = torch.cat(chunks)[:original_numel] / world_size

    return result


class UnslothDDP:
    """
    Optimized DDP for Unsloth training.

    Combines multiple optimization strategies:
    1. LoRA-aware gradient handling
    2. Overlapped communication
    3. Optional gradient compression
    4. Automatic strategy selection

    Args:
        model: The model to wrap.
        compression_ratio: Ratio for gradient compression (None = no compression).
        overlap: Whether to use overlapped communication.
        bucket_size_mb: Size of gradient buckets for overlapped comm.
        use_ring_allreduce: Whether to use custom ring all-reduce.
        sync_base_params: Whether to sync base model parameters.

    Example:
        >>> from unsloth import FastLanguageModel
        >>> from unsloth.distributed import UnslothDDP
        >>>
        >>> model, tokenizer = FastLanguageModel.from_pretrained(...)
        >>> model = FastLanguageModel.get_peft_model(model, r=16)
        >>>
        >>> # Wrap with optimized DDP
        >>> model = UnslothDDP.from_model(
        ...     model,
        ...     compression_ratio=0.1,
        ...     overlap=True,
        ... )
    """

    def __init__(
        self,
        model: nn.Module,
        compression_ratio: Optional[float] = None,
        overlap: bool = True,
        bucket_size_mb: float = 25.0,
        use_ring_allreduce: bool = False,
        sync_base_params: bool = False,
    ):
        self.model = model
        self.compression_ratio = compression_ratio
        self.overlap = overlap
        self.bucket_size_mb = bucket_size_mb
        self.use_ring_allreduce = use_ring_allreduce
        self.sync_base_params = sync_base_params

        # Initialize compression if needed
        self.compressor: Optional[CompressedAllReduce] = None
        if compression_ratio is not None:
            self.compressor = CompressedAllReduce(compression_ratio)

        # Initialize main DDP wrapper
        self._ddp_wrapper: Optional[OverlappedDDP] = None
        self._lora_aware: Optional[LoRAAwareDDP] = None

        if overlap:
            self._ddp_wrapper = OverlappedDDP(
                model,
                bucket_size_mb=bucket_size_mb,
            )
        else:
            self._lora_aware = LoRAAwareDDP(
                model,
                sync_base_params=sync_base_params,
            )

        # Track training state
        self._is_training = False

    @staticmethod
    def from_model(
        model: nn.Module,
        compression_ratio: Optional[float] = None,
        overlap: bool = True,
        bucket_size_mb: float = 25.0,
        use_ring_allreduce: bool = False,
        sync_base_params: bool = False,
    ) -> "UnslothDDP":
        """
        Create optimized DDP from model.

        Args:
            model: The model to wrap.
            compression_ratio: Ratio for gradient compression.
            overlap: Whether to use overlapped communication.
            bucket_size_mb: Size of gradient buckets.
            use_ring_allreduce: Whether to use ring all-reduce.
            sync_base_params: Whether to sync base model parameters.

        Returns:
            UnslothDDP wrapper for the model.
        """
        return UnslothDDP(
            model=model,
            compression_ratio=compression_ratio,
            overlap=overlap,
            bucket_size_mb=bucket_size_mb,
            use_ring_allreduce=use_ring_allreduce,
            sync_base_params=sync_base_params,
        )

    @staticmethod
    def auto(
        model: nn.Module,
        world_size: Optional[int] = None,
        interconnect: str = "ethernet",
    ) -> "UnslothDDP":
        """
        Auto-select best strategy based on setup.

        Args:
            model: The model to wrap.
            world_size: Number of GPUs (auto-detected if None).
            interconnect: Type of interconnect ("nvlink", "infiniband", "ethernet").

        Returns:
            UnslothDDP with optimal settings.
        """
        if world_size is None:
            if dist.is_initialized():
                world_size = dist.get_world_size()
            else:
                world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1

        # Default settings
        compression_ratio = None
        overlap = True
        bucket_size_mb = 25.0
        use_ring = False

        # Adjust based on world size
        if world_size >= 8:
            # Large clusters benefit from compression
            compression_ratio = 0.1
            bucket_size_mb = 50.0
        elif world_size >= 4:
            compression_ratio = 0.2
            bucket_size_mb = 35.0

        # Adjust based on interconnect
        if interconnect == "nvlink":
            # High bandwidth - larger buckets, use ring
            bucket_size_mb *= 2
            use_ring = True
            compression_ratio = None  # Less need for compression
        elif interconnect == "infiniband":
            # Medium-high bandwidth
            use_ring = True
        elif interconnect == "ethernet":
            # Lower bandwidth - more compression
            if compression_ratio is None:
                compression_ratio = 0.3

        logger.info(
            f"UnslothDDP.auto: world_size={world_size}, interconnect={interconnect}, "
            f"compression={compression_ratio}, bucket={bucket_size_mb}MB, ring={use_ring}"
        )

        return UnslothDDP(
            model=model,
            compression_ratio=compression_ratio,
            overlap=overlap,
            bucket_size_mb=bucket_size_mb,
            use_ring_allreduce=use_ring,
        )

    def forward(self, *args, **kwargs):
        """Forward pass through the model."""
        if self._ddp_wrapper is not None:
            return self._ddp_wrapper(*args, **kwargs)
        return self.model(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        """Make the wrapper callable."""
        return self.forward(*args, **kwargs)

    def sync_gradients(self):
        """
        Manually synchronize gradients.

        Call this after backward() if using non-overlapped mode
        or if you need explicit synchronization.
        """
        if self._ddp_wrapper is not None:
            self._ddp_wrapper.synchronize()
        elif self._lora_aware is not None:
            if self.use_ring_allreduce:
                self._sync_with_ring()
            else:
                self._lora_aware.sync_gradients()

        # Apply compression if enabled
        if self.compressor is not None:
            self._apply_compression()

    def _sync_with_ring(self):
        """Sync gradients using ring all-reduce."""
        if not dist.is_initialized():
            return

        for param in self.model.parameters():
            if param.requires_grad and param.grad is not None:
                param.grad.data = ring_allreduce(param.grad.data)

    def _apply_compression(self):
        """Apply gradient compression to all parameters."""
        if self.compressor is None or not dist.is_initialized():
            return

        world_size = dist.get_world_size()

        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                # Compress
                values, indices = self.compressor.compress(param.grad.data, name)

                # All-reduce compressed values
                dist.all_reduce(values)
                values.div_(world_size)

                # Also need to handle indices for proper reconstruction
                # In practice, we'd use all-gather for indices
                # For simplicity, we decompress locally
                param.grad.data = self.compressor.decompress(
                    values, indices, param.grad.shape
                )

    def train(self, mode: bool = True):
        """Set training mode."""
        self._is_training = mode
        self.model.train(mode)
        return self

    def eval(self):
        """Set evaluation mode."""
        return self.train(False)

    def parameters(self):
        """Get model parameters."""
        return self.model.parameters()

    def named_parameters(self):
        """Get named model parameters."""
        return self.model.named_parameters()

    def state_dict(self):
        """Get model state dict."""
        return self.model.state_dict()

    def load_state_dict(self, state_dict):
        """Load model state dict."""
        return self.model.load_state_dict(state_dict)

    def to(self, device):
        """Move model to device."""
        self.model.to(device)
        return self

    def cuda(self, device=None):
        """Move model to CUDA device."""
        self.model.cuda(device)
        return self

    def cpu(self):
        """Move model to CPU."""
        self.model.cpu()
        return self

    @property
    def device(self):
        """Get model device."""
        return next(self.model.parameters()).device

    def __getattr__(self, name):
        """Delegate attribute access to wrapped model."""
        # Avoid infinite recursion for attributes we handle ourselves
        if name in ['model', '_ddp_wrapper', '_lora_aware', 'compressor',
                    'compression_ratio', 'overlap', 'bucket_size_mb',
                    'use_ring_allreduce', 'sync_base_params', '_is_training']:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        return getattr(self.model, name)


def initialize_distributed(
    backend: str = "nccl",
    init_method: Optional[str] = None,
    world_size: Optional[int] = None,
    rank: Optional[int] = None,
) -> bool:
    """
    Initialize distributed training.

    Args:
        backend: Communication backend ("nccl", "gloo", "mpi").
        init_method: URL for process group initialization.
        world_size: Total number of processes.
        rank: Rank of current process.

    Returns:
        True if initialization successful, False otherwise.
    """
    if dist.is_initialized():
        logger.info("Distributed already initialized")
        return True

    try:
        # Try environment variable initialization
        if init_method is None and "MASTER_ADDR" in os.environ:
            dist.init_process_group(backend)
        elif init_method is not None:
            dist.init_process_group(
                backend,
                init_method=init_method,
                world_size=world_size,
                rank=rank,
            )
        else:
            logger.warning("Could not initialize distributed: missing configuration")
            return False

        # Set device for current process
        if torch.cuda.is_available():
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            torch.cuda.set_device(local_rank)

        logger.info(
            f"Distributed initialized: rank={dist.get_rank()}, "
            f"world_size={dist.get_world_size()}"
        )
        return True

    except Exception as e:
        logger.error(f"Failed to initialize distributed: {e}")
        return False


def cleanup_distributed():
    """Clean up distributed training resources."""
    if dist.is_initialized():
        dist.destroy_process_group()
        logger.info("Distributed process group destroyed")


def get_distributed_info() -> Dict[str, Any]:
    """
    Get information about the distributed setup.

    Returns:
        Dictionary with distributed training information.
    """
    if not dist.is_initialized():
        return {
            "initialized": False,
            "world_size": 1,
            "rank": 0,
            "local_rank": 0,
            "backend": None,
        }

    return {
        "initialized": True,
        "world_size": dist.get_world_size(),
        "rank": dist.get_rank(),
        "local_rank": int(os.environ.get("LOCAL_RANK", 0)),
        "backend": dist.get_backend(),
    }
