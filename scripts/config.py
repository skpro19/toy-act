""" Shared architectural constants """

## Transformer (encoder / decoder)
D_MODEL = 512 # input token feat dimensiosn
NUM_IMG_TOKENS = 128
NUM_PROPRIO_TOKENS = 1
NUM_LAYERS = 4


## Transformer Encoder
N_HEAD = 4


## Transformer Decoder
ACTION_CHUNK_SIZE = 10