import torch

from scripts.config import D_MODEL


def get_se(x: torch.Tensor):
    """ Add fixed sinusodial embeddings to the input tensor """
    B, S, C = x.shape
    BASE = 1e4

    # print(f"(B,S,C) => {B,S,C}")

    # PE(pos, 2i)   = sin(pos / 10000^(2i/d))
    # PE(pos, 2i+1) = cos(pos / 10000^(2i/d))

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
    PE = PE.unsqueeze(0).expand(B, -1, -1)

    # print(f"PE.shape => {PE.shape}")

    x = x + PE

    return x


# class SinusodialEmbeddings:
#     def __init__(self, input: torch.Tensor):
#         self.x = input

#         B, S, C = self.x.shape # (batch_size, num_tokens, token_dims)
#         self.pe = torch.zeros(S, C)

#     def get_embeddings(self):
