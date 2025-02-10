import torch
from .clip import CLIP
from .lora import LoRACLIP
from .lora_utils import mark_only_lora_as_trainable

def build_vit_config(state_dict):
    """Extracts Vision Transformer (ViT) configuration from state_dict."""
    vision_width = state_dict["visual.conv1.weight"].shape[0]
    vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
    vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
    grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
    image_resolution = vision_patch_size * grid_size
    return vision_width, vision_layers, vision_patch_size, image_resolution

def build_model(state_dict):
    """Builds the standard CLIP model from a pre-trained state_dict."""
    vision_width, vision_layers, vision_patch_size, image_resolution = build_vit_config(state_dict)

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")))

    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        state_dict.pop(key, None)

    model.load_state_dict(state_dict)
    return model.eval()

def build_LoRA_model(state_dict, r=4, lora_mode="text"):
    """Builds a LoRA-enhanced CLIP model with trainable LoRA layers."""
    vision_width, vision_layers, vision_patch_size, image_resolution = build_vit_config(state_dict)

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")))

    model = LoRACLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers,
        r, lora_mode
    )

    state_dict["lora_text_projection"] = state_dict["text_projection"].T

    res = model.load_state_dict(state_dict, strict=False)
    
    missing_keys = [x for x in res.missing_keys if "lora_" not in x]

    if missing_keys:
        print(f"Warning: Missing keys: {missing_keys}")

    mark_only_lora_as_trainable(model)

    return model.eval()
