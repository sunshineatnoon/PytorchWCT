import numpy as np
from imageio import imread
from scipy.stats import describe
import torch
import torch.nn as nn
from torch.utils.serialization import load_lua
import torchvision.transforms as tvt

import modelsNIPS
import vgg19_normalized
import vgg19_decoders

CHECKPOINT_ENCODER_PY = 'models/vgg19_normalized.pth.tar'
LUA_CHECKPOINT_VGG = 'models/vgg_normalised_conv{}_1.t7'

TEST_IMAGE = 'images/content/in4.jpg'
image_np = imread(TEST_IMAGE).astype(np.float32)

ENCODERS = modelsNIPS.encoder1, modelsNIPS.encoder2, modelsNIPS.encoder3, modelsNIPS.encoder4, modelsNIPS.encoder5
DECODERS = modelsNIPS.decoder1, modelsNIPS.decoder2, modelsNIPS.decoder3, modelsNIPS.decoder4, modelsNIPS.decoder5

# put image into [0, 1], but don't center or normalize like for other nets
trans = tvt.ToTensor()
image_pt = trans(image_np).unsqueeze(0)

def convert_encoder():
  vgg_lua = [load_lua(LUA_CHECKPOINT_VGG.format(k)) for k in range(1, 6)]
  vgg_lua_ = [e(vl) for e, vl in zip(ENCODERS, vgg_lua)]

  vgg_py = vgg19_normalized.VGG19_normalized()

  matching = {
    vgg_py.blocks['conv1_1']: 2,
    vgg_py.blocks['conv1_2']: 5,
    
    vgg_py.blocks['conv2_1']: 9,
    vgg_py.blocks['conv2_2']: 12,

    vgg_py.blocks['conv3_1']: 16,
    vgg_py.blocks['conv3_2']: 19,
    vgg_py.blocks['conv3_3']: 22,
    vgg_py.blocks['conv3_4']: 25,

    vgg_py.blocks['conv4_1']: 29,
    vgg_py.blocks['conv4_2']: 32,
    vgg_py.blocks['conv4_3']: 35,
    vgg_py.blocks['conv4_4']: 38,

    vgg_py.blocks['conv5_1']: 42
  }

  for torch_conv, lua_conv_i in matching.items():
    weights = nn.Parameter(vgg_lua[4].get(lua_conv_i).weight.float())
    bias = nn.Parameter(vgg_lua[4].get(lua_conv_i).bias.float())
    torch_conv.load_state_dict({'weight': weights, 'bias': bias})

  torch.save(vgg_py.state_dict(), CHECKPOINT_ENCODER_PY)

  for k in range(1, 6):
    print(f'encoder {k}')
    e_lua = vgg_lua_[k-1]
    with torch.no_grad():
      al = e_lua(image_pt)
      ap = vgg_py(image_pt, targets=f'relu{k}_1')
    assert al.shape == ap.shape, (al.shape, ap.shape)
    diff = np.abs((al - ap))
    print(describe(diff.flatten()))
    print(np.percentile(diff, 99))
    print()

def convert_decoder(K):
  print(f'converting decoder from layer {K}')
  decoderK_lua = load_lua(f'models/feature_invertor_conv{K}_1.t7')
  decoderK_legacy = DECODERS[K-1](decoderK_lua)
  decoderK_py  = vgg19_decoders.DECODERS[K-1]()

  matching = {
    'conv5_1': -41,

    'conv4_4': -37,
    'conv4_3': -34,
    'conv4_2': -31,
    'conv4_1': -28,

    'conv3_4': -24,
    'conv3_3': -21,
    'conv3_2': -18,
    'conv3_1': -15,
    
    'conv2_2': -11,
    'conv2_1': -8,

    'conv1_2': -4,
    'conv1_1': -1

  }

  for torch_conv, lua_conv_i in matching.items():
    if -lua_conv_i >= len(decoderK_lua):
      continue
    print(f'  {torch_conv}')
    weights = nn.Parameter(decoderK_lua.get(lua_conv_i).weight.float())
    bias = nn.Parameter(decoderK_lua.get(lua_conv_i).bias.float())
    decoderK_py.blocks[torch_conv].load_state_dict({'weight': weights, 'bias': bias})

  torch.save(decoderK_py.state_dict(), f'models/vgg19_normalized_decoder{K}.pth.tar')

  encoder = vgg19_normalized.VGG19_normalized()
  encoder.load_state_dict(torch.load(CHECKPOINT_ENCODER_PY))

  print(f'testing encoding/decoding at layer {K}')

  with torch.no_grad():
    features = encoder(image_pt, targets=f'relu{K}_1')
    rgb_legacy = decoderK_legacy(features)
    rgb_py = decoderK_py(features)
    assert rgb_legacy.shape == rgb_py.shape, (rgb_legacy.shape, rgb_py.shape)
    diff = np.abs((rgb_legacy - rgb_py).numpy())
  print(describe(diff.flatten()))
  print(np.percentile(diff, 99))
  print()

def main():
  convert_encoder()

  for K in range(1, 6):
    convert_decoder(K)

  print('DONE')


if __name__ == '__main__':
  main()
