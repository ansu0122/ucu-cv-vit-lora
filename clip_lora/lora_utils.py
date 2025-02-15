def mark_only_lora_as_trainable(model):
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    print("LoRA parameters are set to trainable. The rest are frozen.")
