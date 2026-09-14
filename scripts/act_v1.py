import torch 
from torch import nn

from transformer_encoder import TransformerEncoder 
from transformer_decoder import TransformerDecoder

from config import (
    D_MODEL, 
    NUM_IMG_TOKENS, 
    NUM_PROPRIO_TOKENS, 
    N_HEAD,
    NUM_LAYERS, 
    ACTION_CHUNK_SIZE
)

class ACTV1(nn.Module):

    def __init__(self, 
                d_model:int = D_MODEL, 
                nhead:int = N_HEAD, 
                num_layers:int = NUM_LAYERS,
                action_chunk_size:int = ACTION_CHUNK_SIZE): 
        
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.k = action_chunk_size
        

        self.img_encoder = nn.Module()
        self.proprio_encoder = nn.Module()
        self.encoder = TransformerEncoder(d_model=self.d_model, nhead=self.nhead, num_layers=self.num_layers)
        self.decoder = TransformerDecoder(d_model=self.d_model, nhead=self.nhead, num_layers=self.num_layers)

        self.decoder_tgt = nn.Parameter(torch.randn(self.k, self.d_model))

    def forward(self, img_tensor: torch.Tensor, proprio_tensor: torch.Tensor):
        
        img_tokens = self.img_encoder(img_tensor) # (B, NUM_IMG_TOKENS, D_MODEL)
        proprio_tokens = self.proprio_encoder(proprio_tensor) # (B, NUM_PROPRIO_TOKENS, D_MODEL)

        # concat img + proprio tokens
        img_proprio_tokens = torch.concat([img_tokens, proprio_tokens], dim=1)

        # encoder => mix image + proprio tokens
        memory = self.encoder(src=img_proprio_tokens)

        # decoder => predict action chunks
        actions = self.decoder(memory=memory, tgt=self.decoder_tgt)
        return actions
        


