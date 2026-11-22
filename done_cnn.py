import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class DoneCNN(nn.Module):
    """A stop model on the full-resolution frame, told the task by the VLM's done-head features.

    ResNet-18 sees detail the VLM's 512 input loses; FiLM from the VLM features picks which
    task's completion it is judging. Max and mean pooling both, so a small cue like a knob
    is not averaged away over the whole frame.
    """

    def __init__(self, feature_dim=960, hidden_dim=256, channels=512):
        super().__init__()
        resnet = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # through layer4, stride 32
        self.film = nn.Linear(feature_dim, 2 * channels)
        self.head = nn.Sequential(nn.Linear(2 * channels, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, images, features):
        """images: uint8 NHWC on the model's device; features: (N, feature_dim). Returns done logits."""
        x = (images.permute(0, 3, 1, 2).float() / 255 - self.mean) / self.std
        with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16):
            maps = self.backbone(x)
            gamma, beta = self.film(features.float()).chunk(2, dim=-1)
            maps = F.relu(maps * (1 + gamma[..., None, None]) + beta[..., None, None])
            pooled = torch.cat([maps.amax(dim=(2, 3)), maps.mean(dim=(2, 3))], dim=-1)
            return self.head(pooled).squeeze(-1).float()

    @torch.no_grad()
    def predict(self, images, features):
        """Done probability for a list of uint8 HWC frames, as Agent.get_action passes them."""
        device = self.mean.device
        batch = torch.from_numpy(np.stack(images)).to(device)
        return torch.sigmoid(self.forward(batch, features.to(device)))
