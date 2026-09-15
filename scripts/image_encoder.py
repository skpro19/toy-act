"""
- image to tokens
- assume single camera setup 
"""

from typing import Tuple

import torch
from torch import nn
from scripts.config import D_MODEL, IMG_DIMS, NUM_IMG_TOKENS
from scripts.embeddings import get_se

class ImageEncoder(nn.Module): 

    def __init__(self, d_model: int = D_MODEL):
        super().__init__()

        self.d_model = d_model
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
        
        self.relu  = nn.ReLU()
        self.project = nn.Linear(in_features=128, out_features=self.d_model)

    def forward(self, x: torch.Tensor):
        
        x = self.conv1(x)
        x = self.relu(x)

        # print(f"x.shape => {x.shape}")

        x = self.conv2(x)
        x = self.relu(x)
        
        # print(f"x.shape => {x.shape}")

        x = self.conv3(x)
        x = self.relu(x)

        # print(f"x.shape => {x.shape}")

        B,C,H,W = x.shape
        # if H * W != NUM_IMG_TOKENS:
        #     raise ValueError(f"H*W != NUM_IMG_TOKENS {H*W} != {NUM_IMG_TOKENS}")


        x = x.permute(0,2,3,1).reshape(B, H*W, C)

        # 128 -> d_model
        x = self.project(x)

        # print(f"x.shape => {x.shape}")

        x = get_se(x)

        # print(f"x.shape => {x.shape}")

        return x