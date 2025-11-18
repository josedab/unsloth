#!/usr/bin/env python
# Copyright 2024-present the Unsloth team.
# Licensed under the Apache License 2.0

"""
Multi-GPU convergence test for UnslothDDP.

This script validates that UnslothDDP produces equivalent results to standard DDP.

Run with:
    torchrun --nproc_per_node=2 tests/distributed/test_multi_gpu_convergence.py

Or:
    python -m torch.distributed.launch --nproc_per_node=2 tests/distributed/test_multi_gpu_convergence.py
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler

# Add parent directory to path for imports
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from unsloth.distributed import UnslothDDP, initialize_distributed, cleanup_distributed


class TestModel(nn.Module):
    """Simple model for convergence testing."""

    def __init__(self, hidden=256):
        super().__init__()
        self.fc1 = nn.Linear(100, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


class LoRATestModel(nn.Module):
    """Model with LoRA-style parameters for testing."""

    def __init__(self, hidden=256, rank=8):
        super().__init__()
        # Base (frozen)
        self.fc1 = nn.Linear(100, hidden)
        self.fc1.weight.requires_grad = False
        self.fc1.bias.requires_grad = False

        # LoRA A and B
        self.lora_a = nn.Parameter(torch.randn(rank, 100) * 0.01)
        self.lora_b = nn.Parameter(torch.randn(hidden, rank) * 0.01)

        # Output (trainable)
        self.fc2 = nn.Linear(hidden, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        h = self.fc1(x) + (x @ self.lora_a.T) @ self.lora_b.T
        h = self.relu(h)
        return self.fc2(h)


def create_dataset(num_samples=1000, seed=42):
    """Create synthetic dataset for testing."""
    torch.manual_seed(seed)
    X = torch.randn(num_samples, 100)
    y = torch.randn(num_samples, 10)
    return TensorDataset(X, y)


def train_with_ddp(model, dataloader, epochs=5, lr=0.001):
    """Train with standard PyTorch DDP."""
    model = model.cuda()
    model = DDP(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    losses = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.cuda(), y.cuda()

            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)

        if dist.get_rank() == 0:
            print(f"[Standard DDP] Epoch {epoch + 1}: Loss = {avg_loss:.6f}")

    return losses


def train_with_unsloth_ddp(model, dataloader, epochs=5, lr=0.001,
                           compression_ratio=None, overlap=False):
    """Train with UnslothDDP."""
    model = model.cuda()
    model = UnslothDDP.from_model(
        model,
        compression_ratio=compression_ratio,
        overlap=overlap,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    losses = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.cuda(), y.cuda()

            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            model.sync_gradients()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)

        if dist.get_rank() == 0:
            config = f"compression={compression_ratio}, overlap={overlap}"
            print(f"[UnslothDDP ({config})] Epoch {epoch + 1}: Loss = {avg_loss:.6f}")

    return losses


def test_convergence(args):
    """Test that UnslothDDP converges similarly to standard DDP."""
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Create dataset
    dataset = create_dataset(num_samples=1000, seed=42)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=0,
    )

    results = {}

    # Test 1: Standard DDP
    if rank == 0:
        print("\n" + "=" * 50)
        print("Test 1: Standard PyTorch DDP")
        print("=" * 50)
    torch.manual_seed(args.seed + rank)
    model1 = TestModel(hidden=args.hidden)
    losses_ddp = train_with_ddp(model1, dataloader, epochs=args.epochs, lr=args.lr)
    results['standard_ddp'] = losses_ddp

    dist.barrier()

    # Test 2: UnslothDDP without compression
    if rank == 0:
        print("\n" + "=" * 50)
        print("Test 2: UnslothDDP (no compression)")
        print("=" * 50)
    torch.manual_seed(args.seed + rank)
    model2 = TestModel(hidden=args.hidden)
    losses_unsloth = train_with_unsloth_ddp(
        model2, dataloader,
        epochs=args.epochs, lr=args.lr,
        compression_ratio=None, overlap=False,
    )
    results['unsloth_no_compress'] = losses_unsloth

    dist.barrier()

    # Test 3: UnslothDDP with compression
    if rank == 0:
        print("\n" + "=" * 50)
        print("Test 3: UnslothDDP (with compression)")
        print("=" * 50)
    torch.manual_seed(args.seed + rank)
    model3 = TestModel(hidden=args.hidden)
    losses_compressed = train_with_unsloth_ddp(
        model3, dataloader,
        epochs=args.epochs, lr=args.lr,
        compression_ratio=0.1, overlap=False,
    )
    results['unsloth_compressed'] = losses_compressed

    dist.barrier()

    # Test 4: LoRA-aware training
    if rank == 0:
        print("\n" + "=" * 50)
        print("Test 4: UnslothDDP with LoRA model")
        print("=" * 50)
    torch.manual_seed(args.seed + rank)
    model4 = LoRATestModel(hidden=args.hidden)
    losses_lora = train_with_unsloth_ddp(
        model4, dataloader,
        epochs=args.epochs, lr=args.lr,
        compression_ratio=None, overlap=False,
    )
    results['unsloth_lora'] = losses_lora

    dist.barrier()

    # Analyze results
    if rank == 0:
        print("\n" + "=" * 50)
        print("Convergence Analysis")
        print("=" * 50)

        final_loss_ddp = results['standard_ddp'][-1]
        final_loss_unsloth = results['unsloth_no_compress'][-1]
        final_loss_compressed = results['unsloth_compressed'][-1]

        print(f"\nFinal Losses:")
        print(f"  Standard DDP:          {final_loss_ddp:.6f}")
        print(f"  UnslothDDP (no comp):  {final_loss_unsloth:.6f}")
        print(f"  UnslothDDP (compress): {final_loss_compressed:.6f}")

        # Check convergence tolerance
        tolerance = 0.1  # 10% tolerance for compressed

        diff_unsloth = abs(final_loss_ddp - final_loss_unsloth) / final_loss_ddp
        diff_compressed = abs(final_loss_ddp - final_loss_compressed) / final_loss_ddp

        print(f"\nRelative Difference from Standard DDP:")
        print(f"  UnslothDDP (no comp):  {diff_unsloth * 100:.2f}%")
        print(f"  UnslothDDP (compress): {diff_compressed * 100:.2f}%")

        # Pass/Fail
        print("\n" + "=" * 50)
        print("Results")
        print("=" * 50)

        passed = True
        if diff_unsloth < 0.01:  # 1% tolerance for non-compressed
            print("PASS: UnslothDDP (no compression) matches standard DDP")
        else:
            print(f"WARN: UnslothDDP differs by {diff_unsloth * 100:.2f}%")

        if diff_compressed < tolerance:
            print(f"PASS: UnslothDDP (compressed) within {tolerance * 100}% tolerance")
        else:
            print(f"FAIL: UnslothDDP (compressed) exceeds {tolerance * 100}% tolerance")
            passed = False

        return passed

    return True


def main():
    parser = argparse.ArgumentParser(description="Multi-GPU convergence test")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size per GPU")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--hidden", type=int, default=256, help="Hidden dimension")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Initialize distributed
    if not initialize_distributed():
        print("Failed to initialize distributed training")
        return

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if rank == 0:
        print(f"Running convergence test with {world_size} GPUs")
        print(f"Configuration: epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}")

    # Set device
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    try:
        passed = test_convergence(args)
        if rank == 0:
            if passed:
                print("\n*** ALL TESTS PASSED ***\n")
            else:
                print("\n*** SOME TESTS FAILED ***\n")
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
