"""
rcdm/dataset.py

A dataset that returns (image_tensor, h) pairs.
Images come from disk; h vectors come from the precomputed .pt file.
The index alignment from precompute_reps.py guarantees they match.
"""

import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


class RepresentationDataset(Dataset):
    """
    Returns (x, h) where:
        x : image tensor (3, image_size, image_size), normalised to [-1, 1]
            — this is what the diffusion model expects
        h : representation tensor (384,) — DinoV3 ViT-S/16 CLS token
            — this is the conditioning vector

    Note on normalisation:
        The encoder uses ImageNet mean/std normalisation.
        The diffusion model expects pixels in [-1, 1] (centre around 0).
        These are two different transforms for two different purposes —
        x uses the diffusion normalisation, h was computed with encoder normalisation.
    """

    def __init__(self, reps_file, image_size=224):
        """
        Args:
            reps_file  : path to the .pt file from precompute_reps.py
                         Supports two formats:
                           - Standard:  {"paths": [...], "reps": Tensor(N,384)}
                             Images are loaded from disk at each __getitem__ call.
                           - Packed:    {"images": Tensor(N,3,H,W) uint8,
                                         "reps":   Tensor(N,384)}
                             Images are stored directly in the file — no disk access
                             needed at training time. Use pack_dataset.py to create.
            image_size : spatial size to resize images to (224 for Messidor-2)
        """
        print(f"Loading representations from {reps_file}...")
        data = torch.load(reps_file, weights_only=False)

        self.reps   = data["reps"]           # Tensor (N, 384)
        self.images = data.get("images")     # Tensor (N, 3, H, W) uint8, or None
        self.paths  = data.get("paths", [])  # list[str], or empty if packed

        if self.images is not None:
            print(f"  {len(self.images)} packed image-representation pairs loaded "
                  f"(no disk access at training time)")
        else:
            print(f"  {len(self.paths)} image-representation pairs loaded "
                  f"(images loaded from disk)")

        self.image_size = image_size

        # Transform for disk-loaded images (PIL → Tensor → [-1, 1])
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
            # Packed format: convert uint8 (0-255) → float32 → [-1, 1]
            x = self.images[idx].float() / 127.5 - 1.0   # (3, H, W) in [-1, 1]
        else:
            # Standard format: load from disk
            img = Image.open(self.paths[idx]).convert("RGB")
            x = self.transform(img)                        # (3, H, W) in [-1, 1]

        h = self.reps[idx]   # (384,) DinoV3 CLS token
        return x, h