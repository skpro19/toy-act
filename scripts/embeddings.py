import torch

from scripts.config import D_MODEL


def get_se_1D(x: torch.Tensor):
    """ Add fixed 1D sinusodial embeddings to the input tensor """
    B, S, C = x.shape
    BASE = 1e4

    # (2i / d)
    freq_scales = torch.arange(0, C, 2, dtype=torch.float32) / C
    freq_scales = torch.pow(torch.tensor(BASE), freq_scales)
    freq_scales = freq_scales.unsqueeze(0).expand(S, -1)

    # pos_indices
    pos_indices = torch.arange(0, S, dtype=torch.float32)
    pos_indices = pos_indices.unsqueeze(0).T
    pos_indices = pos_indices.expand(-1, C // 2)

    # print(f"B.shape => {B.shape}")

    D = pos_indices / freq_scales

    sin_E = torch.sin(D)
    cos_E = torch.cos(D)

    # print(f"sin_E.shape => {sin_E.shape}")
    # print(f"cos_E.shape => {cos_E.shape}")

    PE = torch.zeros(S, C, dtype=torch.float32)
    PE[:, 0::2] = sin_E
    PE[:, 1::2] = cos_E

    # print(f"PE.shape => {PE.shape}")
    PE = PE.unsqueeze(0).expand(B, -1, -1).to(device=x.device, dtype=x.dtype)

    # print(f"PE.shape => {PE.shape}")

    x = x + PE

    return x

def get_se_2D(x: torch.Tensor):
    """ Add fixed 2D sinusodial embeddings to the input tensor """
    B, H, W, C = x.shape
    BASE = 1e4

    x_rows = torch.zeros(1, H, C // 2)
    x_cols = torch.zeros(1, W, C // 2)
    
    re = get_se_1D(x_rows) 
    ce = get_se_1D(x_cols)

    print(f"re.shape => {re.shape}")
    print(f"ce.shape => {ce.shape}")
    
    re = re.expand(W, -1, -1).permute(1,0,2)
    ce = ce.expand(H, -1, -1)

    PE_2D = torch.concat([re, ce], dim=2)
    PE_2D = PE_2D.unsqueeze(0).expand(B, -1, -1, -1).to(device=x.device, dtype=x.dtype)
    
    print(f"PE_2D.shape => {PE_2D.shape}")

    x = x + PE_2D
    return x 