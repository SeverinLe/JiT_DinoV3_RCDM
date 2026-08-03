import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Standard ImageNet normalisation — used by both ResNet-50 and RETFound ViT-Large
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Keys in the MAE pre-training checkpoint that belong to the decoder only.
# These are absent from the ViT fine-tuning model and must be excluded before
# calling load_state_dict(..., strict=False) to avoid spurious warnings.
_MAE_DECODER_PREFIXES = (
    "decoder_embed",
    "decoder_blocks",
    "decoder_norm",
    "decoder_pred",
    "mask_token",
)


# ---------------------------------------------------------------------------
# ResNet-50 encoder (Tiny ImageNet baseline)
# ---------------------------------------------------------------------------

def load_encoder(device="cpu"):
    """
    Load a pretrained ResNet-50 backbone.

    The final classification layer (fc) is replaced with Identity so the
    output is the 2048-dim trunk/backbone representation — the conditioning
    vector h used by RCDM.

    Args:
        device : "cpu" or "cuda"

    Returns:
        encoder : frozen ResNet-50 backbone, eval mode, on device
                  forward(x) → (B, 2048)
    """
    encoder = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    encoder.fc = nn.Identity()  # remove classification head → (B, 2048)

    for param in encoder.parameters():
        param.requires_grad = False

    encoder.eval()
    encoder.to(device)
    return encoder


# ---------------------------------------------------------------------------
# RETFound ViT-Large encoder
# ---------------------------------------------------------------------------

class _RETFoundEncoder(nn.Module):
    """
    Thin wrapper around a timm ViT-Large that:
      - exposes a forward(x) → (B, 1024) CLS-token interface
      - is frozen (no gradients, eval mode)

    Do not construct this directly — use load_retfound_encoder().
    """

    def __init__(self, vit):
        super().__init__()
        self.vit = vit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, 3, 224, 224) ImageNet-normalised tensor
        Returns:
            h : (B, 1024) CLS-token representation
        """
        # forward_features returns the full token sequence (B, 197, 1024).
        # Token 0 is the CLS token — the global image representation.
        tokens = self.vit.forward_features(x)   # (B, 197, 1024)
        return tokens[:, 0]                      # (B, 1024)


def load_retfound_encoder(weights_path: str, device: str = "cpu") -> nn.Module:
    """
    Load the frozen RETFound ViT-Large MAE encoder.

    RETFound is a ViT-Large (patch 16, 224 px) pre-trained with Masked
    Autoencoders on 1.6 M retinal images.  The .pth checkpoint stores both
    the encoder and the MAE decoder; we discard the decoder and return only
    the frozen encoder that maps an image to a 1024-dim CLS-token vector.

    Architecture (from data/RETFound/config.json):
        hidden_size   : 1024
        num_heads     : 16
        num_layers    : 24
        image_size    : 224
        patch_size    : 16

    Args:
        weights_path : path to the .pth checkpoint
                       (e.g. "data/RETFound/RETFound_mae_natureOCT.pth")
        device       : "cpu" or "cuda"

    Returns:
        encoder : frozen _RETFoundEncoder, eval mode, on device
                  forward(x) → (B, 1024)
    """
    try:
        import timm
    except ImportError:
        raise ImportError(
            "timm is required for the RETFound encoder.  "
            "Install it with:  pip install timm"
        )

    print(f"  Building ViT-Large/16 (timm)...")
    # num_classes=0  → no classification head; forward_features returns all tokens
    # global_pool="" → do not pool — we extract the CLS token ourselves so we
    #                  have explicit control over which token is used
    vit = timm.create_model(
        "vit_large_patch16_224",
        pretrained=False,
        num_classes=0,
        global_pool="",
    )

    print(f"  Loading checkpoint from {weights_path} ...")
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model", checkpoint)

    # Drop MAE decoder keys — they have no matching layers in the ViT encoder
    encoder_state = {
        k: v for k, v in state_dict.items()
        if not k.startswith(_MAE_DECODER_PREFIXES)
    }

    msg = vit.load_state_dict(encoder_state, strict=False)
    n_missing    = len(msg.missing_keys)
    n_unexpected = len(msg.unexpected_keys)
    print(f"  Weights loaded — missing: {n_missing}, unexpected: {n_unexpected}")
    if n_missing:
        print(f"    missing : {msg.missing_keys[:5]}{'...' if n_missing > 5 else ''}")
    if n_unexpected:
        print(f"    unexpected: {msg.unexpected_keys[:5]}{'...' if n_unexpected > 5 else ''}")

    for param in vit.parameters():
        param.requires_grad = False

    vit.eval()

    encoder = _RETFoundEncoder(vit)
    encoder.eval().to(device)
    return encoder


# ---------------------------------------------------------------------------
# Shared preprocessing
# ---------------------------------------------------------------------------

def build_transform(image_size: int = 64) -> transforms.Compose:
    """
    Preprocessing pipeline for any input image before encoding.

    Both ResNet-50 and RETFound ViT-Large use standard ImageNet mean/std
    normalisation.  Pass image_size=64 for Tiny ImageNet, image_size=224
    for RETFound.

    Returns a Compose transform:  PIL Image → (3, image_size, image_size)
    """
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ---------------------------------------------------------------------------
# Encoding helpers (encoder-agnostic)
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_image(image_path: str, encoder: nn.Module,
                 transform: transforms.Compose, device: str = "cpu") -> torch.Tensor:
    """
    Extract the backbone representation from a single image file.

    Works with both load_encoder() (ResNet-50 → 2048-dim) and
    load_retfound_encoder() (ViT-Large → 1024-dim).

    Args:
        image_path : path to image file (jpg, png, anything PIL can open)
        encoder    : frozen encoder returned by load_encoder() or
                     load_retfound_encoder()
        transform  : preprocessing pipeline from build_transform()
        device     : must match the encoder's device

    Returns:
        h : torch.Tensor of shape (1, D)
    """
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    return encoder(x)


@torch.no_grad()
def encode_batch(image_paths: list, encoder: nn.Module,
                 transform: transforms.Compose,
                 device: str = "cpu", batch_size: int = 64) -> torch.Tensor:
    """
    Extract representations for a list of image paths.

    Encoder-agnostic — works with ResNet-50 (D=2048) and
    RETFound ViT-Large (D=1024).

    Used by precompute_reps.py to cache all training representations before
    training starts, avoiding repeated encoder forward passes during training.

    Args:
        image_paths : list of file paths (str)
        encoder     : frozen encoder from load_encoder() or
                      load_retfound_encoder()
        transform   : from build_transform()
        device      : must match the encoder's device
        batch_size  : images per forward pass — reduce if OOM

    Returns:
        reps : torch.Tensor of shape (N, D)
    """
    all_reps = []

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]

        imgs = []
        for p in batch_paths:
            img = Image.open(p).convert("RGB")
            imgs.append(transform(img))

        x = torch.stack(imgs).to(device)
        h = encoder(x)
        all_reps.append(h.cpu())

        if i % (batch_size * 10) == 0:
            print(f"  encoded {i}/{len(image_paths)} images")

    return torch.cat(all_reps, dim=0)