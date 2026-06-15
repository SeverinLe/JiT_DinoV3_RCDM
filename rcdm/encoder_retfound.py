"""
rcdm/encoder_retfound.py

Frozen SSL encoder for JiT-RCDM: loads RETFound (MAE ViT-Large/16, fine-tuned
on retinal CFP images) and extracts CLS tokens.

Origin / changes vs rcdm/encoder.py (DinoV3 ViT-S/16)
------------------------------------------------------
  This module is a drop-in alternative encoder, mirroring rcdm/encoder.py's
  structure and function signatures exactly so it can be swapped into
  precompute_reps.py / sampling.py without touching the rest of the pipeline
  (rcdm/jit.py, rcdm/conditioning.py, rcdm/dataset.py, scripts/train.py are
  already parametrised by h_dim).

  What changed vs DinoV3 ViT-S/16:
    - Encoder        : DinoV3 ViT-S/16 (HuggingFace AutoModel) → RETFound
                        MAE ViT-Large/16 (timm vit_large_patch16_224).
    - CLS-token dim  : 384 → 1024.
    - Checkpoint     : local HuggingFace dir (config.json + safetensors) →
                        a single MAE training-save .pth containing
                        {"model": state_dict, "optimizer": ..., "args": ...}.
    - CLS extraction : out.last_hidden_state[:, 0, :] (HF API)
                        → encoder.forward_features(x)[:, 0, :] (timm API).
    - Normalisation  : unchanged. RETFound's official training pipeline
                        (RETFound_MAE/util/datasets.py) uses
                        timm.data.constants.IMAGENET_DEFAULT_MEAN/STD, the
                        same ImageNet mean/std already used for DinoV3.

  Why filter the checkpoint's state dict:
    RETFound_mae_natureCFP.pth is a raw MAE *pretraining* checkpoint: it
    contains both the ViT-L encoder (cls_token, pos_embed, patch_embed,
    blocks.0-23, norm) AND the MAE decoder (every "decoder_*" key —
    decoder_embed, decoder_pos_embed, decoder_blocks, decoder_norm,
    decoder_pred — plus mask_token). The decoder is only used to reconstruct
    masked patches during pretraining and has no counterpart in timm's
    vit_large_patch16_224 (num_classes=0, no decoder). Loading with
    strict=False after dropping the decoder/mask keys leaves zero
    missing/unexpected keys for the encoder itself.
"""

import os
import torch
import timm
from torchvision import transforms
from PIL import Image

# ── path to the local RETFound MAE ViT-L/16 checkpoint ──
RETFOUND_CHECKPOINT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "checkpoints", "retfound", "RETFound_mae_natureCFP.pth",
)

# CLS-token dimension for RETFound MAE ViT-Large/16
# Changed from rcdm/encoder.py: 384 (DinoV3 ViT-S/16 CLS token) → 1024 (ViT-L/16 CLS token)
ENCODER_OUTPUT_DIM = 1024

# Prefixes of MAE-decoder-only keys present in the pretraining checkpoint but
# absent from timm's encoder-only vit_large_patch16_224 — must be dropped
# before load_state_dict.
_MAE_DECODER_PREFIXES = (
    "decoder_",
    "mask_token",
)


def load_encoder(device="cpu", checkpoint_path=RETFOUND_CHECKPOINT):
    """
    Load RETFound MAE ViT-Large/16 as the frozen encoder.

    rcdm/encoder.py equivalent:
        encoder = AutoModel.from_pretrained(local_path)  # HF DinoV3, CLS token output
    Our version:
        encoder = timm.create_model("vit_large_patch16_224", num_classes=0)
        encoder.load_state_dict(filtered_checkpoint["model"], strict=False)

    Freezing is explicit: requires_grad=False + eval() ensures the encoder
    never updates during training and Dropout is in inference mode.

    Args:
        device          : "cpu", "cuda", or "mps"
        checkpoint_path : path to the RETFound MAE .pth training save
                          (must contain a top-level "model" state dict)

    Returns:
        encoder : frozen RETFound ViT-Large/16 backbone in eval mode on device
    """
    encoder = timm.create_model("vit_large_patch16_224", pretrained=False, num_classes=0)

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["model"]

    # ── drop MAE-decoder-only keys — timm's encoder has no decoder ──
    filtered = {
        k: v for k, v in state_dict.items()
        if not k.startswith(_MAE_DECODER_PREFIXES)
    }

    missing, unexpected = encoder.load_state_dict(filtered, strict=False)
    if missing or unexpected:
        print(f"  [encoder_retfound] missing keys: {missing}")
        print(f"  [encoder_retfound] unexpected keys: {unexpected}")

    # ── Freeze all encoder parameters — encoder is a fixed feature extractor ──
    for param in encoder.parameters():
        param.requires_grad = False

    encoder.eval()
    encoder.to(device)
    return encoder


# ── ImageNet normalisation constants ──
# Same as rcdm/encoder.py: RETFound_MAE's official training pipeline
# (util/datasets.py) uses timm.data.constants.IMAGENET_DEFAULT_MEAN/STD,
# which equal these values.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def build_transform(image_size=224):
    """
    Preprocessing pipeline for RETFound MAE ViT-Large/16.

    Always produces 224x224 output regardless of what image_size is passed.
    RETFound ViT-L/16 has a fixed positional embedding grid of 14x14 patches
    (224 / 16 = 14) — same constraint as DinoV3 ViT-S/16 in rcdm/encoder.py.

    rcdm/encoder.py equivalent: identical structure, RETFound just shares the
    same ImageNet normalisation constants.
    """
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


@torch.no_grad()
def encode_image(image_path, encoder, transform, device="cpu"):
    """
    Extract the 1024-dim CLS token from a single image via RETFound.

    Output shape: (1, 1024) — encoder.forward_features(x)[:, 0, :].
    Index 0 is the CLS token; indices 1-196 are the 196 patch tokens
    (not used here).

    rcdm/encoder.py equivalent:
        out = encoder(pixel_values=x)
        h   = out.last_hidden_state[:, 0, :]   # (1, 384)
    """
    img = Image.open(image_path).convert("RGB")
    x   = transform(img).unsqueeze(0).to(device)   # (1, 3, 224, 224)
    feats = encoder.forward_features(x)
    h   = feats[:, 0, :]                            # CLS token → (1, 1024)
    return h


@torch.no_grad()
def encode_batch(image_paths, encoder, transform, device="cpu", batch_size=64):
    """
    Extract representations for a list of image paths in batches.

    Args:
        image_paths : list of file paths (length N)
        encoder     : frozen RETFound ViT-L/16 from load_encoder()
        transform   : from build_transform(224)
        device      : must match encoder's device
        batch_size  : images per forward pass (reduce if OOM — ViT-L/16 is
                      much larger than DinoV3 ViT-S/16)

    Returns:
        reps : Tensor (N, 1024) — one CLS token per image
    """
    all_reps = []

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]

        imgs = []
        for p in batch_paths:
            img = Image.open(p).convert("RGB")
            imgs.append(transform(img))

        x     = torch.stack(imgs).to(device)        # (B, 3, 224, 224)
        feats = encoder.forward_features(x)
        h     = feats[:, 0, :]                        # CLS token → (B, 1024)
        all_reps.append(h.cpu())

        if i % (batch_size * 10) == 0:
            print(f"  encoded {i}/{len(image_paths)} images")

    return torch.cat(all_reps, dim=0)              # (N, 1024)
