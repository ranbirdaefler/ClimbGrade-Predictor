"""DINOv2 hold embedding wrapper -- loads model once, reuses for all holds."""

from __future__ import annotations

import torch
from PIL import Image
from torchvision import transforms


class HoldEmbedder:
    """Wraps DINOv2 for hold-crop embedding extraction."""

    def __init__(self, model_name: str = "dinov2_vitb14", device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = torch.hub.load("facebookresearch/dinov2", model_name)
        self.model = self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def embed_single(self, crop_pil: Image.Image) -> torch.Tensor:
        """Embed one crop. Returns shape [768]."""
        t = self.transform(crop_pil).unsqueeze(0).to(self.device)
        return self.model(t).squeeze(0)

    @torch.no_grad()
    def embed_batch(self, crops: list[Image.Image]) -> torch.Tensor:
        """Embed a list of crops. Returns shape [N, 768]."""
        tensors = torch.stack([self.transform(c) for c in crops]).to(self.device)
        return self.model(tensors)
