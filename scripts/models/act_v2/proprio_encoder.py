""" Generate proprio tokens embeddings """

from torch import nn 
import torch 
from scripts.models.act_v1.config import D_MODEL, PROPRIO_DIMS

class ProprioEncoder(nn.Module): 

    def __init__(self, d_model:int = D_MODEL, proprio_dims: int = PROPRIO_DIMS):
        
        super().__init__()
        self.d_model = d_model
        self.proprio_dims = proprio_dims
        self.project = nn.Linear(self.proprio_dims, self.d_model)

    def forward(self, x:torch.Tensor):
        
        if x.ndim != 3:
            raise ValueError(f"x.ndim != 3, x.ndim={x.ndim}")

        out = self.project(x)
        return out
