import os
import torch
import urllib.request
from clip_lora.model_builder import build_LoRA_model

_MODELS = {
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
}

def download_clip_weights(model_name="ViT-B/16", save_path="models/clip_weights.pt"):
    """Download CLIP weights and save locally."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if model_name not in _MODELS:
        raise ValueError(f"Model {model_name} not found. Available models: {list(_MODELS.keys())}")

    url = _MODELS[model_name]
    print(f"Downloading {model_name} from {url}...")
    urllib.request.urlretrieve(url, save_path)
    print(f"Saved CLIP weights to {save_path}")
    model = torch.jit.load(save_path, map_location="cpu")
    state_dict = model.state_dict()

    return state_dict


def load_clip_with_lora(model_name="ViT-B/32", r=4, lora_mode="text", save_path="models/clip_weights.pt"):
    """Loads CLIP model, adds LoRA layers, and returns a LoRA-CLIP model."""
    state_dict = download_clip_weights(model_name, save_path)
    model = build_LoRA_model(state_dict, r=r, lora_mode=lora_mode)
    return model


if __name__ == "__main__":

    lora_clip = load_clip_with_lora("ViT-B/16", r=8, lora_mode="vision")

    
    print(lora_clip)
    print("LoRA-CLIP model loaded successfully!")