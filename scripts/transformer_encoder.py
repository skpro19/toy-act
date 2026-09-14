"""Self-attention over image + proprio tokens"""

from torch import nn
import torch

from config import D_MODEL, N_HEAD, NUM_LAYERS

class TransformerEncoder(nn.Module):

    def __init__(self, 
                d_model:int = D_MODEL, 
                nhead:int = N_HEAD, 
                num_layers:int = NUM_LAYERS) -> None:
            
        super().__init__()
        
        self.d_model = d_model # token dim
        self.nhead = nhead
        self.num_layers = num_layers
        
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model,
                                                        nhead=self.nhead,
                                                        dim_feedforward=4 * self.d_model,
                                                        dropout=0.1,
                                                        batch_first=True, 
                                                        norm_first=True)

        self.encoder = nn.TransformerEncoder(encoder_layer=self.encoder_layer, num_layers=num_layers)
    
    def forward(self, src: torch.Tensor):
        out = self.encoder(src)
        return out

        