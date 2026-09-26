from torch import nn
import torch
from scripts.models.act_v2.cvae_encoder import CVAEEncoder
from scripts.models.act_v2.proprio_encoder import ProprioEncoder
from scripts.models.act_v2.action_encoder import ActionEncoder
from scripts.models.act_v2.z_encoder import ZEncoder
from scripts.models.act_v2.image_encoder import ImageEncoder
from scripts.models.act_v2.transformer_encoder import TransformerEncoder
from scripts.models.act_v2.transformer_decoder import TransformerDecoder

from scripts.models.act_v2.config import (
    D_MODEL,
    N_HEAD, 
    NUM_LAYERS, 
    Z_DIMS,
    PROPRIO_DIMS, 
    ACTION_CHUNK_SIZE
)

from scripts.models.act_v2.embeddings import get_se_1D
from scripts.models.act_v2.z_encoder import ZEncoder

class ACTV2(nn.Module):

    def __init__(self, 
                d_model:int =D_MODEL, 
                nhead:int =N_HEAD, 
                num_layers:int =NUM_LAYERS,
                z_dims:int = Z_DIMS,
                proprio_dims:int = PROPRIO_DIMS, 
                action_chunk_size:int = ACTION_CHUNK_SIZE,
                use_z: bool = True):

        super().__init__()

        self.use_z = use_z
        self.d_model = d_model 
        self.nhead = nhead
        self.num_layers = num_layers
        self.z_dims = z_dims
        self.proprio_dims = proprio_dims
        self.k = action_chunk_size
        
        self.cvae_encoder = CVAEEncoder(d_model=self.d_model, 
                                        nhead=self.nhead, 
                                        num_layers=self.num_layers, 
                                        z_dims=self.z_dims)

        self.proprio_encoder    = ProprioEncoder(d_model=self.d_model, proprio_dims=self.proprio_dims)
        self.action_encoder     = ActionEncoder(d_model=self.d_model, action_dims=self.proprio_dims)
        self.z_encoder          = ZEncoder(d_model=self.d_model, z_dims=self.z_dims)
        self.image_encoder      = ImageEncoder(d_model=self.d_model)


        # cvae decoder
        self.transformer_encoder = TransformerEncoder(d_model=self.d_model, 
                                                    nhead=self.nhead,
                                                    num_layers=self.num_layers)

        self.transformer_decoder = TransformerDecoder(d_model=self.d_model, 
                                                    nhead=self.nhead, 
                                                    num_layers=self.num_layers)
        
        self.cls = nn.Parameter(torch.randn(1,self.d_model))
        self.action_head = nn.Linear(self.d_model, self.proprio_dims)

    def posterior(self, *, proprio: torch.Tensor, actions: torch.Tensor):
        """CVAE posterior q(z | proprio, action_chunk): returns (mu, log_sigma_x2)."""
        B, _, _ = proprio.shape

        cls = self.cls.unsqueeze(0).expand(B, -1, -1)
        proprio_tokens = self.proprio_encoder(proprio)
        action_tokens = self.action_encoder(actions)

        src_cvae_encoder = torch.concat([cls, proprio_tokens, action_tokens], dim=1)

        return self.cvae_encoder(src=src_cvae_encoder)

    def decode_from_tokens(
        self,
        *,
        img_tokens: torch.Tensor,
        proprio_tokens: torch.Tensor,
        z: torch.Tensor | None = None,
        use_z: bool | None = None,
    ) -> torch.Tensor:
        """Decode an action chunk from pre-encoded image/proprio tokens."""
        if use_z is None:
            use_z = self.use_z

        if use_z:
            if z is None:
                raise ValueError("z is required when use_z is True")
            z_token = self.z_encoder(z)
            src_transformer_encoder = torch.concat([img_tokens, proprio_tokens, z_token], dim=1)
        else:
            src_transformer_encoder = torch.concat([img_tokens, proprio_tokens], dim=1)

        memory = self.transformer_encoder(src_transformer_encoder)

        B = proprio_tokens.shape[0]
        decoder_tgt = get_se_1D(
            torch.zeros(B, self.k, self.d_model, dtype=memory.dtype, device=memory.device)
        )

        decoded_actions = self.transformer_decoder(tgt=decoder_tgt, memory=memory)

        return self.action_head(decoded_actions)

    def decode(
        self,
        *,
        proprio: torch.Tensor,
        img: torch.Tensor,
        z: torch.Tensor | None = None,
        use_z: bool | None = None,
    ) -> torch.Tensor:
        """Decode an action chunk for an observation."""
        proprio_tokens = self.proprio_encoder(proprio)
        img_tokens = self.image_encoder(img)

        return self.decode_from_tokens(
            img_tokens=img_tokens,
            proprio_tokens=proprio_tokens,
            z=z,
            use_z=use_z,
        )

    def forward(self, proprio: torch.Tensor, actions: torch.Tensor, img: torch.Tensor):

        proprio_tokens = self.proprio_encoder(proprio)
        img_tokens = self.image_encoder(img)

        # mu, log(sigma-squared)
        mu, log_sigma_x2 = self.posterior(proprio=proprio, actions=actions)

        z = None
        if self.use_z:
            z = mu + torch.randn_like(mu) * torch.sqrt(torch.exp(log_sigma_x2))

        actions = self.decode_from_tokens(
            img_tokens=img_tokens,
            proprio_tokens=proprio_tokens,
            z=z,
            use_z=self.use_z,
        )

        return actions, mu, log_sigma_x2

    @torch.no_grad()
    def infer(self, proprio: torch.Tensor, img: torch.Tensor):
        """Decode actions at inference time (prior z when use_z is enabled)."""
        z = None
        if self.use_z:
            B, _, _ = proprio.shape
            z = torch.zeros(B, 1, self.z_dims, device=proprio.device, dtype=proprio.dtype)

        return self.decode(proprio=proprio, img=img, z=z, use_z=self.use_z)
