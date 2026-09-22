import torch 
from torch import nn 
from scripts.models.act_v2.config import (
    D_MODEL,
    PROPRIO_DIMS
)
from scripts.models.act_v2.embeddings import get_se_1D

class ActionEncoder(nn.Module): 

    def __init__(self, d_model:int = D_MODEL, action_dims:int = PROPRIO_DIMS):
        
        super().__init__()

        self.d_model = d_model
        self.action_dims = action_dims

        self.encoder = nn.Linear(self.action_dims, self.d_model)
        # self.embeddings = get_se_1D()

    def forward(self, actions:torch.Tensor):
        out = self.encoder(actions)
        out = get_se_1D(out)

        return out
