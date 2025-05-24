import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiResolutionTokenizer(nn.Module):
    def __init__(self, input_dim, embedding_dim, kernel_sizes=[3, 7, 15], strides=[1, 2, 4]):
        super().__init__()
        self.tokenizers = nn.ModuleList()
        
        for k, s in zip(kernel_sizes, strides):
            self.tokenizers.append(nn.Sequential(
                nn.Conv1d(input_dim, embedding_dim // len(kernel_sizes), kernel_size=k, stride=s, padding=k//2),
                nn.GELU()
            ))
    
    def forward(self, x):
        tokens = []
        for tokenizer in self.tokenizers:
            token = tokenizer(x)
            tokens.append(token)
        
        min_length = min(t.size(-1) for t in tokens)
        aligned_tokens = []
        
        for t in tokens:
            if t.size(-1) > min_length:
                aligned_t = F.interpolate(t, size=min_length, mode='linear', align_corners=False)
                aligned_tokens.append(aligned_t)
            else:
                aligned_tokens.append(t)
        
        return torch.cat(aligned_tokens, dim=1)
