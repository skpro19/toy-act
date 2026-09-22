""" Shared architectural constants """

## Image Encoder
NUM_IMG_TOKENS = 128
IMG_DIMS = (84,84)

## Proprio Encoder
NUM_PROPRIO_TOKENS = 1
JOINT_DIMS = 7
PROPRIO_DIMS = 8

## Transformer (encoder / decoder)
D_MODEL = 512 # input token feat dimensiosn
NUM_LAYERS = 4


## Transformer Encoder
N_HEAD = 4


## Transformer Decoder
ACTION_CHUNK_SIZE = 10

## CVAE 
Z_DIMS = 32
