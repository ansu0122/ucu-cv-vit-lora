import torch
from torch import nn
from collections import OrderedDict
from .loralib import LoRALinear as lora_Linear
from .loralib import LoRAEmbedding as lora_Embedding
from .loralib import LoRAConv2d as lora_Conv2d
from .loralib import LoRAMultiheadAttention as lora_MultiheadAttention
from .vision import VisionTransformer
from .text import Transformer

class LoRAVisionTransformer(nn.Module):
    """LoRA-enabled Vision Transformer (ViT) for CLIP."""
    def __init__(self, input_resolution, patch_size, width, layers, heads, output_dim, r=4):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim

        self.conv1 = lora_Conv2d(3, width, kernel_size=patch_size, stride=patch_size, bias=False, r=r)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = nn.LayerNorm(width)

        self.transformer = LoRATransformer(width, layers, heads, r=r)

        self.ln_post = nn.LayerNorm(width)
        self.proj = lora_Linear(width, output_dim, r=r)

    def forward(self, x):
        x = self.conv1(x)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat([self.class_embedding.expand(x.shape[0], 1, -1), x], dim=1)
        x = x + self.positional_embedding
        x = self.ln_pre(x)
        x = self.transformer(x)
        x = self.ln_post(x[:, 0, :])
        return self.proj(x)

class LoRAResidualAttentionBlock(nn.Module):
    """LoRA-enhanced Transformer block with multi-head attention."""
    def __init__(self, d_model, n_head, attn_mask=None, r=4):
        super().__init__()

        self.attn = lora_MultiheadAttention(d_model, n_head, r=r)
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", lora_Linear(d_model, d_model * 4, r=r)),
            ("gelu", nn.GELU()),
            ("c_proj", lora_Linear(d_model * 4, d_model, r=r))
        ]))
        self.ln_2 = nn.LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class LoRATransformer(nn.Module):
    """LoRA-enabled Transformer with residual attention blocks."""
    def __init__(self, width, layers, heads, attn_mask=None, r=4):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[LoRAResidualAttentionBlock(width, heads, attn_mask, r=r) for _ in range(layers)])

    def forward(self, x):
        return self.resblocks(x)

class LoRACLIP(nn.Module):
    """LoRA-enhanced CLIP model with modular text and vision processing."""
    def __init__(self, embed_dim, image_resolution, vision_layers, vision_width, vision_patch_size,
                 context_length, vocab_size, transformer_width, transformer_heads, transformer_layers, r=4, lora_mode="text"):
        super().__init__()

        self.context_length = context_length

        if "vision" in lora_mode:
            self.visual = LoRAVisionTransformer(
                image_resolution, vision_patch_size, vision_width, vision_layers, 
                vision_width // 64, embed_dim, r=r
            )
        else:
            self.visual = VisionTransformer(
                image_resolution, vision_patch_size, vision_width, vision_layers, 
                vision_width // 64, embed_dim
            )

        if "text" in lora_mode:
            self.transformer = LoRATransformer(transformer_width, transformer_layers, transformer_heads, r=r)
            self.token_embedding = lora_Embedding(vocab_size, transformer_width, r=r)
            self.lora_text_projection = lora_Linear(transformer_width, embed_dim, r=r, bias=False)
        else:
            self.transformer = Transformer(transformer_width, transformer_layers, transformer_heads)
            self.token_embedding = nn.Embedding(vocab_size, transformer_width)
            self.lora_text_projection = nn.Linear(transformer_width, embed_dim, bias=False)

        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = nn.LayerNorm(transformer_width)
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / 0.07)))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

    def encode_text(self, text):
        x = self.token_embedding(text)
        x = x + self.positional_embedding
        x = self.ln_final(self.transformer(x))
        return self.lora_text_projection(x)

    def encode_image(self, image):
        return self.visual(image)

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.T
        logits_per_text = logits_per_image.T

        return logits_per_image, logits_per_text
