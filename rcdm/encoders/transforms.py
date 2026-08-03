"""
rcdm/encoders/transforms.py

Preprocessing shared by every frozen encoder in this package.

All three encoders (DinoV3 ViT-S/16, RETFound MAE ViT-L/16, ResNet-50) consume
ImageNet-normalised input.  RETFound's official pipeline
(RETFound_MAE/util/datasets.py) uses timm's IMAGENET_DEFAULT_MEAN/STD, which
are the same constants; DinoV3 and torchvision's ResNet-50 use them too.

Note the deliberate separation of two normalisations in this project:

    encoder input   ImageNet mean/std      (this module)
    generator input [-1, 1]                (rcdm/dataset.py)

They are never mixed.  h is computed in encoder space; the diffusion target x
lives in [-1, 1].
"""

from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Every supported encoder is a /16 ViT (or a ResNet trained at this resolution)
# with a fixed 14x14 positional-embedding grid: 224 / 16 = 14 patches per side.
ENCODER_INPUT_SIZE = 224


def build_transform(image_size: int = ENCODER_INPUT_SIZE) -> transforms.Compose:
    """
    PIL image -> (3, image_size, image_size) ImageNet-normalised tensor.

    Args:
        image_size: edge length in pixels.  Leave at the default 224 unless you
            know what you are doing — a ViT's positional embeddings are tied to
            the 14x14 grid, and any other resolution silently interpolates or
            corrupts them.  The *generator's* resolution is a separate setting
            and does not constrain this one.

    Returns:
        A torchvision Compose: Resize -> CenterCrop -> ToTensor -> Normalize.
    """
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
