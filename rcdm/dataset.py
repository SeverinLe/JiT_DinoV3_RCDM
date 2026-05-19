"""
rcdm/dataset.py

Dataset returning (image_tensor, h) pairs for JiT-RCDM training.

Origin / changes vs upstream repos
------------------------------------
  FROM RCDM (adapted).

  RCDM's dataset loaded images from disk on every __getitem__ call and paired
  them with representations computed on-the-fly by the encoder. Our version
  reads precomputed representations from a .pt file (aligned by index from
  precompute_reps.py) and supports two file formats:

  Changes made:
    1. h_dim 2048 → 384: RCDM stored (N, 2048) ResNet-50 avgpool vectors.
       We store (N, 384) DinoV3 ViT-S/16 CLS tokens.

    2. Precomputed reps: RCDM ran the encoder at training time. We precompute
       once (scripts/precompute_reps.py) and load from disk. The encoder never
       runs during training.

    3. Dual-format support (NEW): added packed format for Colab compatibility.
       RCDM only had the path-based format. The packed format embeds uint8 image
       tensors directly in the .pt file — eliminates absolute path dependency
       when training on a different machine (e.g. Google Colab).

    4. Normalisation split (clarified): x uses [-1, 1] diffusion normalisation
       (same as RCDM). h was computed with ImageNet normalisation (encoder
       convention). RCDM used the same normalisation for both; we keep them
       separate because the encoder and the generator have different conventions.
"""

import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

# ── FROM RCDM — same ImageNet constants ──
# Used only for the disk-loading path; packed images use a direct uint8 conversion.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


class RepresentationDataset(Dataset):
    """
    Returns (x, h) pairs for flow-matching training.

        x : (3, image_size, image_size)  float32 in [-1, 1]  — generator input/target
        h : (384,)                        float32             — DinoV3 CLS conditioning

    Supports two .pt file formats (detected automatically):

      Standard (RCDM-style):  {"paths": [...], "reps": Tensor(N, 384)}
        Images are loaded from disk at each __getitem__ call.
        Requires the absolute paths stored at precompute time to still be valid.

      Packed (new):           {"images": Tensor(N, 3, H, W) uint8, "reps": Tensor(N, 384)}
        Images are stored as uint8 tensors directly in the file (~150 MB for 972 images).
        No disk I/O at training time. Required for Colab where local paths don't exist.
        Created by scripts/pack_dataset.py.

    Normalisation note:
        x is normalised to [-1, 1] — the diffusion model's convention.
        h was computed with ImageNet mean/std normalisation — the encoder's convention.
        These are independent and must not be mixed.
    """

    def __init__(self, reps_file, image_size=224):
        """
        Args:
            reps_file  : path to the .pt file produced by precompute_reps.py
                         (standard) or pack_dataset.py (packed).
            image_size : spatial size for disk-loaded images (ignored for packed,
                         which uses the stored tensor dimensions directly).
        """
        print(f"Loading representations from {reps_file}...")
        data = torch.load(reps_file, weights_only=False)

        # ── changed from RCDM: reps are (N, 384) DinoV3 CLS tokens, not (N, 2048) ResNet ──
        self.reps = data["reps"]               # Tensor (N, 384)

        # ── NEW: packed format — images embedded as uint8 tensors ──
        # RCDM only had path-based loading. Packed format added for Colab.
        self.images = data.get("images")       # Tensor (N, 3, H, W) uint8, or None
        self.paths  = data.get("paths", [])    # list[str], or empty if packed

        if self.images is not None:
            print(f"  {len(self.images)} packed image-representation pairs loaded "
                  f"(no disk access at training time)")
        else:
            print(f"  {len(self.paths)} image-representation pairs loaded "
                  f"(images loaded from disk)")

        self.image_size = image_size

        # ── FROM RCDM (same structure): PIL → Tensor → [-1, 1] for disk-loaded images ──
        # Normalise(0.5, 0.5) maps [0,1] → [-1, 1]: the diffusion model's convention.
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.images) if self.images is not None else len(self.paths)

    def __getitem__(self, idx):
        if self.images is not None:
            # ── NEW (packed format): uint8 [0, 255] → float32 → [-1, 1] ──
            # uint8 / 127.5 - 1.0 is equivalent to (pixel/255 - 0.5) / 0.5
            # Quantisation error < 1/255 ≈ 0.004 in [-1,1] space — negligible for MSE.
            x = self.images[idx].float() / 127.5 - 1.0    # (3, H, W) in [-1, 1]
        else:
            # ── FROM RCDM: load image from disk at __getitem__ time ──
            img = Image.open(self.paths[idx]).convert("RGB")
            x   = self.transform(img)                       # (3, H, W) in [-1, 1]

        # ── changed from RCDM: h is (384,) DinoV3 CLS token, not (2048,) ResNet avgpool ──
        h = self.reps[idx]
        return x, h
