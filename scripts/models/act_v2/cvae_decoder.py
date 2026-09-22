import torch 
from torch import nn

from scripts.models.act_v1.transformer_encoder import TransformerEncoder
from scripts.models.act_v1.transformer_decoder import TransformerDecoder
from scripts.models.act_v1.image_encoder import ImageEncoder
from scripts.models.act_v1.proprio_encoder import ProprioEncoder
from scripts.models.act_v1.embeddings import get_se_1D

from scripts.models.act_v1.config import (
    D_MODEL, 
    NUM_IMG_TOKENS, 
    NUM_PROPRIO_TOKENS, 
    N_HEAD,
    NUM_LAYERS, 
    ACTION_CHUNK_SIZE,
    PROPRIO_DIMS
)

class ACTV1(nn.Module):

    def __init__(self, 
                d_model:int = D_MODEL, 
                nhead:int = N_HEAD, 
                num_layers:int = NUM_LAYERS,
                action_chunk_size:int = ACTION_CHUNK_SIZE,
                proprio_dims:int = PROPRIO_DIMS):

        super().__init__()
        
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.k = action_chunk_size
        self.proprio_dims = proprio_dims

        self.img_encoder = ImageEncoder(d_model=self.d_model)
        self.proprio_encoder = ProprioEncoder(d_model=self.d_model, proprio_dims=self.proprio_dims)
        self.encoder =  TransformerEncoder(d_model=self.d_model, nhead=self.nhead, num_layers=self.num_layers)
        self.decoder = TransformerDecoder(d_model=self.d_model, nhead=self.nhead, num_layers=self.num_layers)

        # self.decoder_tgt = nn.Parameter(torch.randn(self.k, self.d_model))
        # self.decoder_tgt = get_se(x = torch.zeros(1, self.k, self.d_model))

        self.action_head = nn.Linear(self.d_model, self.proprio_dims)

    def forward(self, img_tensor: torch.Tensor, proprio_tensor: torch.Tensor):
        
        img_tokens = self.img_encoder(img_tensor) # (B, NUM_IMG_TOKENS, D_MODEL)
        proprio_tokens = self.proprio_encoder(proprio_tensor) # (B, NUM_PROPRIO_TOKENS, D_MODEL)

        # concat img + proprio tokens
        img_proprio_tokens = torch.concat([img_tokens, proprio_tokens], dim=1)

        # encoder => mix image + proprio tokens
        memory = self.encoder(src=img_proprio_tokens)

        B, _, _ = memory.shape
        decoder_tgt = get_se_1D(torch.zeros(B, self.k, self.d_model, dtype=memory.dtype, device=memory.device))

        # print(f"decoder_tgt.shape => {decoder_tgt.shape}")
        # decoder_tgt = decoder_tgt.unsqueeze(0).expand(B, -1, -1)

        # decoder => predict action chunks
        decoded_actions = self.decoder(memory=memory, tgt=decoder_tgt)

        actions = self.action_head(decoded_actions)

        return actions
        
