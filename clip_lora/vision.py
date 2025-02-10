import torch
from torch import nn
from .text import Transformer

class VisionTransformer(nn.Module):
    """Standard Vision Transformer (ViT) for CLIP."""
    def __init__(self, input_resolution, patch_size, width, layers, heads, output_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(3, width, kernel_size=patch_size, stride=patch_size, bias=False)
        self.class_embedding = nn.Parameter(torch.randn(width))
        self.positional_embedding = nn.Parameter(torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = nn.LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = nn.LayerNorm(width)
        self.proj = nn.Parameter(torch.randn(width, output_dim))

    def forward(self, x):
        x = self.conv1(x)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat([self.class_embedding.expand(x.shape[0], 1, -1), x], dim=1)
        x = x + self.positional_embedding
        x = self.ln_pre(x)
        x = self.transformer(x)
        x = self.ln_post(x[:, 0, :])
        return x @ self.proj
