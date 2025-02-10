import torch
from torch import nn

class LoRALinear(nn.Module):
    """LoRA-enabled Linear layer"""
    def __init__(self, in_features, out_features, r=4, bias=True):
        super().__init__()
        self.r = r
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        if r > 0:
            self.lora_a = nn.Linear(in_features, r, bias=False)
            self.lora_b = nn.Linear(r, out_features, bias=False)
        else:
            self.lora_a, self.lora_b = None, None

    def forward(self, x):
        if self.r > 0:
            return self.linear(x) + self.lora_b(self.lora_a(x))
        return self.linear(x)

class LoRAEmbedding(nn.Embedding):
    """LoRA-enabled Embedding layer"""
    def __init__(self, num_embeddings, embedding_dim, r=4):
        super().__init__(num_embeddings, embedding_dim)
        self.r = r
        if r > 0:
            self.lora_a = nn.Parameter(torch.zeros(num_embeddings, r))
            self.lora_b = nn.Parameter(torch.zeros(r, embedding_dim))
            nn.init.xavier_uniform_(self.lora_a)
            nn.init.xavier_uniform_(self.lora_b)

    def forward(self, x):
        return super().forward(x) + torch.matmul(self.lora_a, self.lora_b)[x]

class LoRAConv2d(nn.Conv2d):
    """LoRA-enabled Conv2d layer"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, r=4):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.r = r
        if r > 0:
            self.lora_a = nn.Conv2d(in_channels, r, kernel_size, stride, padding, bias=False)
            self.lora_b = nn.Conv2d(r, out_channels, 1, bias=False)

    def forward(self, x):
        if self.r > 0:
            return super().forward(x) + self.lora_b(self.lora_a(x))
        return super().forward(x)

class LoRAMultiheadAttention(nn.MultiheadAttention):
    """LoRA-enabled Multihead Attention layer"""
    def __init__(self, embed_dim, num_heads, r=4, **kwargs):
        super().__init__(embed_dim, num_heads, **kwargs)
        self.r = r
        if r > 0:
            self.lora_q = nn.Linear(embed_dim, r, bias=False)
            self.lora_k = nn.Linear(embed_dim, r, bias=False)
            self.lora_v = nn.Linear(embed_dim, r, bias=False)
            self.lora_out = nn.Linear(r, embed_dim, bias=False)

    def forward(self, query, key, value, **kwargs):
        if self.r > 0:
            query = query + self.lora_out(self.lora_q(query))
            key = key + self.lora_out(self.lora_k(key))
            value = value + self.lora_out(self.lora_v(value))
        return super().forward(query, key, value, **kwargs)
