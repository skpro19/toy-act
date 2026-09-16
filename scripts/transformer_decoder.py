"""
Policy action-chunk transformer (Figure 2 RIGHT, inner decoder).

Cross-attention from k fixed query slots to the observation encoder output.
DETR-style — not BERT (no [CLS] token here).
"""

from torch import nn
import torch
from scripts.config import D_MODEL, N_HEAD, NUM_LAYERS

class TransformerDecoder(nn.Module):

    def __init__(self, 
                d_model:int = D_MODEL, 
                nhead:int  = N_HEAD, 
                num_layers:int = NUM_LAYERS) -> None:
        
        super().__init__()
        self.d_model = d_model 
        self.nhead = nhead
        self.num_layers = num_layers

        self.decoder_layer = nn.TransformerDecoderLayer(d_model=self.d_model,
                                                        nhead = self.nhead,
                                                        dim_feedforward=4 * self.d_model, 
                                                        dropout=0.1, 
                                                        batch_first=True,
                                                        norm_first=True)

        self.decoder = nn.TransformerDecoder(decoder_layer=self.decoder_layer, num_layers=num_layers)

    def forward(self, memory:torch.Tensor, tgt:torch.Tensor):
        
        out = self.decoder(tgt=tgt, memory=memory)
        return out
