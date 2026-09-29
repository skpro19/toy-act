"""
Encode RGB views into tokens for the ACT policy transformer encoder.

Each camera view passes through the same conv stack (shared weights). Input is
5D: (batch, num_cameras, 3, height, width). Output is
(batch, num_cameras * tokens_per_view, d_model).
"""

from typing import Tuple

import torch
from torch import nn
from scripts.models.act_v1.config import D_MODEL, IMG_DIMS, NUM_IMG_TOKENS
from scripts.models.act_v1.embeddings import get_se_2D

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
        
        B,N,C,H,W = x.shape

        x  = x.reshape(B * N, C, H, W)
        x = self.conv1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.relu(x)
        
        x = self.conv3(x)
        x = self.relu(x)

        # B,C,H,W = x.shape
        x = x.permute(0,2,3,1)

        
        # [C =>  d_model] projection
        x = self.project(x)

        
        # enrich x with 2D positional sinusodial embeddings         
        x = get_se_2D(x)
        
        # 2D => 1D tokens
        # x = x.reshape(B, H * W, self.d_model)
        # x = x.reshape(B // N , N * H * W, self.d_model)
        x = x.reshape(B , -1 , self.d_model)
        # print(f"x.shape => {x.shape}")

        return x
