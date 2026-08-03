"""
scripts/train.py

Train RCDM on a dataset using precomputed encoder representations.

Usage — Tiny ImageNet (ResNet-50, h_dim=2048):
    python scripts/train.py \
        --reps_file   data/tiny-imagenet-200/train_reps.pt \
        --save_dir    checkpoints/tiny-imagenet \
        --image_size  64 \
        --h_dim       2048 \
        --num_channels 128 \
        --batch_size  128 \
        --lr          1e-4 \
        --total_steps 50000 \
        --save_interval 2000 \
        --log_interval  100 \
        --num_workers 4 \
        --device cuda

Usage — OCT2017 with RETFound (ViT-Large, h_dim=1024):
    python scripts/train.py \
        --reps_file   data/OCT2017/train_reps.pt \
        --save_dir    checkpoints/retfound-oct \
        --image_size  224 \
        --h_dim       1024 \
        --num_channels 192 \
        --batch_size  32 \
        --lr          1e-4 \
        --total_steps 100000 \
        --save_interval 2000 \
        --log_interval  100 \
        --num_workers 4 \
        --device cuda
"""

import argparse
import sys
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parent.parent))

from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
)
from guided_diffusion.resample import create_named_schedule_sampler
from rcdm.dataset import RepresentationDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps_file",      type=str,   default="data/tiny-imagenet-200/train_reps.pt")
    parser.add_argument("--save_dir",       type=str,   default="checkpoints/")
    parser.add_argument("--image_size",     type=int,   default=64,
                        help="Spatial size of training images (64 for Tiny ImageNet, 224 for RETFound)")
    parser.add_argument("--h_dim",          type=int,   default=2048,
                        help="Dimension of the conditioning representation vector "
                             "(2048 for ResNet-50, 1024 for RETFound ViT-Large). "
                             "Auto-detected from the reps file if not set.")
    parser.add_argument("--num_channels",   type=int,   default=128,
                        help="Base channel width of the UNet "
                             "(128 for 64px images, 192–256 for 224px images)")
    parser.add_argument("--batch_size",     type=int,   default=16)
    parser.add_argument("--lr",             type=float, default=1e-4)
    parser.add_argument("--total_steps",    type=int,   default=100000)
    parser.add_argument("--save_interval",  type=int,   default=5000)
    parser.add_argument("--log_interval",   type=int,   default=100)
    parser.add_argument("--device",         type=str,   default="cpu")
    parser.add_argument("--num_workers",    type=int,   default=0,
                        help="DataLoader worker processes (0=main process; use 4 on Colab/GPU)")
    parser.add_argument("--resume",         type=str,   default=None,
                        help="Path to checkpoint to resume from")
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.save_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 1 — Build model and diffusion process
    # ------------------------------------------------------------------ #
    print("\n[1/4] Building model...")

    # Auto-detect h_dim from the reps file if it records the encoder metadata.
    # This avoids silent mismatches when switching between ResNet-50 and RETFound.
    reps_meta = torch.load(args.reps_file, map_location="cpu", weights_only=False)
    if "h_dim" in reps_meta:
        detected_h_dim = reps_meta["h_dim"]
        if detected_h_dim != args.h_dim:
            print(f"  NOTE: --h_dim {args.h_dim} overridden by reps file metadata "
                  f"→ h_dim={detected_h_dim}")
            args.h_dim = detected_h_dim
    del reps_meta  # free memory before loading the full dataset below

    model_args = model_and_diffusion_defaults()
    model_args.update({
        "image_size":           args.image_size,
        "num_channels":         args.num_channels,
        "num_res_blocks":       2,
        "learn_sigma":          True,
        "diffusion_steps":      1000,
        "noise_schedule":       "linear",
        "use_scale_shift_norm": True,
        "h_dim":                args.h_dim,
    })
    print(f"  image_size={args.image_size}, h_dim={args.h_dim}, "
          f"num_channels={args.num_channels}")

    model, diffusion = create_model_and_diffusion(**model_args)
    model.to(device)

    # Resume from checkpoint if provided
    if args.resume:
        print(f"  Resuming from {args.resume}")
        state = torch.load(args.resume, map_location=device)
        model.load_state_dict(state["model"])
        start_step = state["step"]
        print(f"  Resuming from step {start_step}")
    else:
        start_step = 0

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {total_params / 1e6:.1f}M")

    # ------------------------------------------------------------------ #
    # 2 — Dataset and dataloader
    # ------------------------------------------------------------------ #
    print("\n[2/4] Loading dataset...")
    dataset = RepresentationDataset(
        reps_file=args.reps_file,
        image_size=args.image_size,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(args.device == "cuda"),
        drop_last=True,
    )
    print(f"  {len(dataset)} samples, "
          f"{len(dataloader)} batches per epoch at batch_size={args.batch_size}")

    # ------------------------------------------------------------------ #
    # 3 — Optimiser and timestep sampler
    # ------------------------------------------------------------------ #
    print("\n[3/4] Setting up optimiser...")
    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.0,
    )

    # Uniform timestep sampler — samples t uniformly from [0, T]
    # This is the standard approach from the ADM paper
    schedule_sampler = create_named_schedule_sampler("uniform", diffusion)

    # ------------------------------------------------------------------ #
    # 4 — Training loop
    # ------------------------------------------------------------------ #
    print("\n[4/4] Starting training...\n")
    model.train()

    step       = start_step
    data_iter  = iter(dataloader)
    total_loss = 0.0

    while step < args.total_steps:

        # Get next batch — restart iterator when dataset is exhausted
        try:
            x, h = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            x, h = next(data_iter)

        x = x.to(device)   # (B, 3, H, H)   images in [-1, 1]
        h = h.to(device)   # (B, h_dim)      encoder representations

        # Sample a random timestep for each image in the batch
        # t shape: (B,)   weights shape: (B,) — used for importance sampling
        t, weights = schedule_sampler.sample(x.shape[0], device)

        # Ask diffusion to compute the training loss
        # Internally this:
        #   1. adds noise to x according to t  →  noisy_x
        #   2. runs model(noisy_x, t, h=h)
        #   3. computes MSE between predicted and actual noise
        losses = diffusion.training_losses(
            model=model,
            x_start=x,
            t=t,
            model_kwargs={"h": h},   # passed as **kwargs to model.forward()
        )

        # losses["loss"] is shape (B,) — one loss value per sample
        # weight by importance sampling weights, then average
        loss = (losses["loss"] * weights).mean()

        optimiser.zero_grad()
        loss.backward()

        # Gradient clipping — prevents occasional large gradient spikes
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimiser.step()

        total_loss += loss.item()
        step += 1

        # ---- logging ----
        if step % args.log_interval == 0:
            avg_loss = total_loss / args.log_interval
            print(f"step {step:>7d}/{args.total_steps} | loss {avg_loss:.4f}")
            total_loss = 0.0

        # ---- checkpointing ----
        if step % args.save_interval == 0:
            ckpt_path = Path(args.save_dir) / f"rcdm_step{step:07d}.pt"
            torch.save(
                {
                    "step":       step,
                    "model":      model.state_dict(),
                    "optimiser":  optimiser.state_dict(),
                    "model_args": model_args,
                },
                ckpt_path,
            )
            print(f"  → saved checkpoint: {ckpt_path}")

    # Save final model
    final_path = Path(args.save_dir) / "rcdm_final.pt"
    torch.save(
        {
            "step":       step,
            "model":      model.state_dict(),
            "optimiser":  optimiser.state_dict(),
            "model_args": model_args,
        },
        final_path,
    )
    print(f"\nTraining complete. Final model saved to {final_path}")


if __name__ == "__main__":
    main()