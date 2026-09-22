import torch  
from torch import nn 

from scripts.models.act_v2.config import Z_DIMS, D_MODEL

class ZEncoder(nn.Module): 

    def __init__(self, z_dims:int = Z_DIMS, d_model:int = D_MODEL): 
        
        super().__init__() 
        self.d_model = d_model 
        self.z_dims = z_dims
        
        self.encoder = nn.Linear(self.z_dims, self.d_model)

    def forward(self, z: torch.Tensor):
        out = self.encoder(z)
        return out 