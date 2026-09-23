from torch import nn 
import torch 
from scripts.models.act_v2.config import (
    ACTION_CHUNK_SIZE, 
    D_MODEL, 
    N_HEAD, 
    NUM_LAYERS, 
    Z_DIMS)

"""  
    CVAE Encoder => leans phi_theta(z | proprio , action_chunks) 
    input => [cls] + proprio + action-chunks 
    output => z
"""



class CVAEEncoder(nn.Module):

    def __init__(self, 
                d_model:int = D_MODEL,
                nhead:int = N_HEAD,
                num_layers:int = NUM_LAYERS,
                z_dims:int = Z_DIMS):
        
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.z_dims = z_dims
        
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=D_MODEL,
                                                        nhead = nhead,
                                                        dim_feedforward=4 * self.d_model,
                                                        dropout=0.1,
                                                        batch_first=True, 
                                                        norm_first=True)
                                                         
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=self.encoder_layer, 
                                                num_layers=self.num_layers)

        # predict mean + sigma-squared
        self.z_head = nn.Linear(self.d_model, 2 * self.z_dims)


    def forward(self, src:torch.Tensor):
        
        """ src => k action chunks + proprio_token  + cls token """
        
        out = self.transformer_encoder(src) # [B, k + 2, d_model]

        # print(f"[cvae encoder] out.shape => {out.shape}")

        # cls_out = out[:, 0, :]

        # print(f"cls_out.shape => {cls_out.shape}")

        cls_out = self.z_head(out[:, 0, :])
        cls_out = cls_out.unsqueeze(1)
        # print(f"cls_out.shape => {cls_out.shape}")
        # print(f"z_out.shape => {z_out.shape}")

        mu = cls_out[..., :self.z_dims]
        log_sigma_x2 = cls_out[..., self.z_dims:]

        # print(f"[cvae_encoder] mu.shape => {mu.shape} log_sigma_x2.shape => {log_sigma_x2.shape}")

        # print(f"mu.shape => {mu.shape}")
        # print(f"log_sigma_x2.shape => {log_sigma_x2.shape}")
        
        return mu, log_sigma_x2