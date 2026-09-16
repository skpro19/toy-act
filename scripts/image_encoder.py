"""
- image to tokens
- assume single camera setup 
"""

from typing import Tuple

import torch
from torch import nn
from scripts.config import D_MODEL, IMG_DIMS, NUM_IMG_TOKENS
from scripts.embeddings import  get_se_2D

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

        x = x.permute(0,2,3,1)

        print(f"[permute] x.shape => {x.shape}")

        # [C =>  d_model] projection
        x = self.project(x)

        print(f"[project] x.shape => {x.shape}")

        # enrich x with 2D positional sinusodial embeddings         
        x = get_se_2D(x)
        print(f"x.shape => {x.shape}")

        # 2D => 1D tokens
        x = x.reshape(B, H * W, self.d_model)
        print(f"x.shape => {x.shape}")

        return x