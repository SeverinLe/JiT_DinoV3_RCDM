"""
rcdm/encoder.py

Frozen SSL encoder for JiT-RCDM: loads DinoV3 ViT-S/16 and extracts CLS tokens.

Origin / changes vs upstream repos
------------------------------------
  ENTIRELY NEW FILE — not present in RCDM or JiT.

  RCDM loaded its encoder (torchvision ResNet-50) inline inside train.py with
  no dedicated module. There was no precomputation — the encoder ran on every
  training step.

  JiT had no encoder at all: it was class-conditional (integer labels), so no
  image encoder was needed.

  What we added:
    - load_encoder()  : loads DinoV3 ViT-S/16 from a local HuggingFace checkpoint,
                        freezes all weights, sets eval() mode.
    - build_transform(): ImageNet-normalised preprocessing at fixed 224×224.
                        Always 224 px — DinoV3 has a fixed 14×14 positional grid
                        (224 / 16 = 14 patches per side). Using any other resolution
                        would silently corrupt the positional embeddings.
    - encode_image()  : single-image encoding utility.
    - encode_batch()  : batched encoding used by precompute_reps.py to cache all
                        training representations before training starts.

  Why DinoV3 ViT-S/16 over RCDM's ResNet-50:
    - ResNet-50 avgpool encodes classification pressure, not perceptual similarity.
      DINO-style SSL learns representations where nearby vectors → perceptually
      similar images, which is what representation-conditioned generation requires.
    - DinoV3 is fine-tuned on medical/retinal data → CLS token is domain-aligned
      with Messidor-2 (vessel topology, disc morphology, pathology stage).
    - ViT-S/16 (384-dim) is compact vs ResNet-50 (2048-dim), keeping the
      conditioning path lightweight.

  Why local checkpoint (not HuggingFace hub):
    - Reproducibility: weights are fixed, not tied to a remote model version.
    - Offline training (Colab, air-gapped clusters).
    - Allows the critical config.json fix (use_gated_mlp: false) without
      touching a shared remote model.

  use_gated_mlp fix (critical):
    The local checkpoint was trained with a standard 2-proj FFN (up_proj +
    down_proj), but config.json shipped with use_gated_mlp: true, which expects
    a 3-proj gated FFN. HuggingFace would randomly initialise the missing
    gate_proj on every load → non-deterministic h vectors between training runs
    and inference. Fix: set "use_gated_mlp": false in config.json.
"""

import os
import torch
from torchvision import transforms
from transformers import AutoModel
from PIL import Image

# ── NEW: path to the local DinoV3 ViT-S/16 checkpoint ──
# RCDM loaded torchvision.models.resnet50(pretrained=True) — no local checkpoint needed.
# We load from a local HuggingFace-format directory for reproducibility and offline use.
DINOV3_CHECKPOINT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "checkpoints", "dinov3_vits16_tmp",
)

# CLS-token dimension for DinoV3 ViT-S/16
# Changed from RCDM: 2048 (ResNet-50 avgpool) → 384 (ViT-S/16 CLS token)
ENCODER_OUTPUT_DIM = 384


def load_encoder(device="cpu", checkpoint_path=DINOV3_CHECKPOINT):
    """
    Load DinoV3 ViT-S/16 from a local HuggingFace checkpoint as the frozen encoder.

    RCDM equivalent:
        model = torchvision.models.resnet50(pretrained=True)
        encoder = nn.Sequential(*list(model.children())[:-1])  # avgpool output
    Our version:
        encoder = AutoModel.from_pretrained(local_path)  # CLS token output

    Freezing is explicit: requires_grad=False + eval() ensures the encoder
    never updates during training and BatchNorm/Dropout are in inference mode.

    Args:
        device          : "cpu", "cuda", or "mps"
        checkpoint_path : path to HuggingFace checkpoint dir
                          (must contain config.json + model.safetensors)

    Returns:
        encoder : frozen DinoV3-S backbone in eval mode on device
    """
    encoder = AutoModel.from_pretrained(checkpoint_path, local_files_only=True)

    # ── Freeze all encoder parameters — encoder is a fixed feature extractor ──
    # RCDM also froze its encoder; the pattern is the same, just applied here.
    for param in encoder.parameters():
        param.requires_grad = False

    encoder.eval()
    encoder.to(device)
    return encoder


# ── NEW: ImageNet normalisation constants ──
# DinoV3 was trained with ImageNet mean/std — same as DinoV2.
# RCDM also used ImageNet normalisation for its ResNet-50 encoder.
# Note: these are the ENCODER's normalisation constants, not the diffusion model's.
# The diffusion model uses [-1, 1] (centre 0.5, scale 0.5) — a different transform.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def build_transform(image_size=224):
    """
    Preprocessing pipeline for DinoV3 ViT-S/16.

    Always produces 224×224 output regardless of what image_size is passed.
    DinoV3 ViT-S/16 has a fixed positional embedding grid of 14×14 patches
    (224 / 16 = 14). Any other resolution would silently corrupt the
    positional information — this is different from the generator's image_size,
    which controls output resolution and can be set independently.

    RCDM equivalent:
        transforms.Resize(64), CenterCrop(64), ToTensor(), Normalize(ImageNet)
    Our version:
        same structure, resolution fixed to 224 for the encoder.
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
    Extract the 384-dim CLS token from a single image via DinoV3.

    Output shape: (1, 384) — the CLS token at last_hidden_state[:, 0, :].
    Index 0 is always CLS. Indices 1–4 are register tokens (ignored).
    Indices 5–200 are the 196 patch tokens (not used here).

    RCDM equivalent:
        with torch.no_grad():
            h = encoder(x)   # ResNet avgpool → (1, 2048, 1, 1) → squeeze → (1, 2048)
    """
    img = Image.open(image_path).convert("RGB")
    x   = transform(img).unsqueeze(0).to(device)   # (1, 3, 224, 224)
    out = encoder(pixel_values=x)
    h   = out.last_hidden_state[:, 0, :]            # CLS token → (1, 384)
    return h


@torch.no_grad()
def encode_batch(image_paths, encoder, transform, device="cpu", batch_size=64):
    """
    Extract representations for a list of image paths in batches.

    NEW: RCDM ran the encoder on every training step (online encoding).
    We precompute all representations once and cache them to disk via
    precompute_reps.py. This decouples encoder speed from training speed
    and ensures representations are consistent across all training steps.

    Args:
        image_paths : list of file paths (length N)
        encoder     : frozen DinoV3-S from load_encoder()
        transform   : from build_transform(224)
        device      : must match encoder's device
        batch_size  : images per forward pass (reduce if OOM)

    Returns:
        reps : Tensor (N, 384) — one CLS token per image
    """
    all_reps = []

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]

        imgs = []
        for p in batch_paths:
            img = Image.open(p).convert("RGB")
            imgs.append(transform(img))

        x   = torch.stack(imgs).to(device)         # (B, 3, 224, 224)
        out = encoder(pixel_values=x)
        h   = out.last_hidden_state[:, 0, :]        # CLS token → (B, 384)
        all_reps.append(h.cpu())

        if i % (batch_size * 10) == 0:
            print(f"  encoded {i}/{len(image_paths)} images")

    return torch.cat(all_reps, dim=0)              # (N, 384)
