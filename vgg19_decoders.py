import torch
import torch.nn as nn
from collections import OrderedDict

class VGG19Decoder1(nn.Module):

  def __init__(self):

    super(VGG19Decoder1, self).__init__()

    # input shape originally 224 x 224

    self.blocks = OrderedDict([  # {{{
      ('pad1_1', nn.ReflectionPad2d(1)),  # 226 x 226
      ('conv1_1', nn.Conv2d(64, 3, 3, 1, 0)),  # 224 x 224
    ])  # }}}

    self.seq = nn.Sequential(self.blocks)


  def forward(self, x, targets=None):
    return self.seq(x)

class VGG19Decoder2(nn.Module):

  def __init__(self):

    super(VGG19Decoder2, self).__init__()

    # input shape originally 224 x 224

    self.blocks = OrderedDict([  # {{{
      ('pad2_1',  nn.ReflectionPad2d(1)),# {{{}}}
      ('conv2_1', nn.Conv2d(128, 64, 3, 1, 0)),
      ('relu2_1', nn.ReLU(inplace=True)),  # 112 x 112

      ('unpool1', nn.Upsample(scale_factor=2)),  # 112 x 112
      ('pad1_2', nn.ReflectionPad2d(1)),
      ('conv1_2', nn.Conv2d(64, 64, 3, 1, 0)),
      ('relu1_2', nn.ReLU(inplace=True)),  # 224 x 224
      ('pad1_1', nn.ReflectionPad2d(1)),  # 226 x 226
      ('conv1_1', nn.Conv2d(64, 3, 3, 1, 0)),  # 224 x 224
    ])  # }}}

    self.seq = nn.Sequential(self.blocks)


  def forward(self, x, targets=None):
    return self.seq(x)

class VGG19Decoder3(nn.Module):

  def __init__(self):

    super(VGG19Decoder3, self).__init__()

    # input shape originally 224 x 224

    self.blocks = OrderedDict([  # {{{
      ('pad3_1',  nn.ReflectionPad2d(1)),
      ('conv3_1', nn.Conv2d(256, 128, 3, 1, 0)),
      ('relu3_1', nn.ReLU(inplace=True)),  # 56 x 56

      ('unpool2', nn.Upsample(scale_factor=2)),  # 56 x 56
      ('pad2_2',  nn.ReflectionPad2d(1)),
      ('conv2_2', nn.Conv2d(128, 128, 3, 1, 0)),
      ('relu2_2', nn.ReLU(inplace=True)),  # 112 x 112
      ('pad2_1',  nn.ReflectionPad2d(1)),# {{{}}}
      ('conv2_1', nn.Conv2d(128, 64, 3, 1, 0)),
      ('relu2_1', nn.ReLU(inplace=True)),  # 112 x 112

      ('unpool1', nn.Upsample(scale_factor=2)),  # 112 x 112
      ('pad1_2', nn.ReflectionPad2d(1)),
      ('conv1_2', nn.Conv2d(64, 64, 3, 1, 0)),
      ('relu1_2', nn.ReLU(inplace=True)),  # 224 x 224
      ('pad1_1', nn.ReflectionPad2d(1)),  # 226 x 226
      ('conv1_1', nn.Conv2d(64, 3, 3, 1, 0)),  # 224 x 224
    ])  # }}}

    self.seq = nn.Sequential(self.blocks)


  def forward(self, x, targets=None):
    return self.seq(x)

class VGG19Decoder4(nn.Module):

  def __init__(self):

    super(VGG19Decoder4, self).__init__()

    # input shape originally 224 x 224

    self.blocks = OrderedDict([  # {{{
      ('pad4_1',  nn.ReflectionPad2d(1)),
      ('conv4_1', nn.Conv2d(512, 256, 3, 1, 0)),
      ('relu4_1', nn.ReLU(inplace=True)),  # 28 x 28

      ('unpool3', nn.Upsample(scale_factor=2)),  # 28 x 28
      ('pad3_4',  nn.ReflectionPad2d(1)),
      ('conv3_4', nn.Conv2d(256, 256, 3, 1, 0)),
      ('relu3_4', nn.ReLU(inplace=True)),  # 56 x 56
      ('pad3_3',  nn.ReflectionPad2d(1)),
      ('conv3_3', nn.Conv2d(256, 256, 3, 1, 0)),
      ('relu3_3', nn.ReLU(inplace=True)),  # 56 x 56
      ('pad3_2',  nn.ReflectionPad2d(1)),
      ('conv3_2', nn.Conv2d(256, 256, 3, 1, 0)),
      ('relu3_2', nn.ReLU(inplace=True)),  # 56 x 56
      ('pad3_1',  nn.ReflectionPad2d(1)),
      ('conv3_1', nn.Conv2d(256, 128, 3, 1, 0)),
      ('relu3_1', nn.ReLU(inplace=True)),  # 56 x 56

      ('unpool2', nn.Upsample(scale_factor=2)),  # 56 x 56
      ('pad2_2',  nn.ReflectionPad2d(1)),
      ('conv2_2', nn.Conv2d(128, 128, 3, 1, 0)),
      ('relu2_2', nn.ReLU(inplace=True)),  # 112 x 112
      ('pad2_1',  nn.ReflectionPad2d(1)),# {{{}}}
      ('conv2_1', nn.Conv2d(128, 64, 3, 1, 0)),
      ('relu2_1', nn.ReLU(inplace=True)),  # 112 x 112

      ('unpool1', nn.Upsample(scale_factor=2)),  # 112 x 112
      ('pad1_2', nn.ReflectionPad2d(1)),
      ('conv1_2', nn.Conv2d(64, 64, 3, 1, 0)),
      ('relu1_2', nn.ReLU(inplace=True)),  # 224 x 224
      ('pad1_1', nn.ReflectionPad2d(1)),  # 226 x 226
      ('conv1_1', nn.Conv2d(64, 3, 3, 1, 0)),  # 224 x 224
    ])  # }}}

    self.seq = nn.Sequential(self.blocks)


  def forward(self, x, targets=None):
    return self.seq(x)

class VGG19Decoder5(nn.Module):

  def __init__(self):

    super(VGG19Decoder5, self).__init__()

    # input shape originally 224 x 224

    self.blocks = OrderedDict([  # {{{
      ('pad5_1',  nn.ReflectionPad2d(1)),
      ('conv5_1', nn.Conv2d(512, 512, 3, 1, 0)),
      ('relu5_1', nn.ReLU(inplace=True)),  # 14 x 14

      ('unpool4', nn.Upsample(scale_factor=2)),
      ('pad4_4',  nn.ReflectionPad2d(1)),
      ('conv4_4', nn.Conv2d(512, 512, 3, 1, 0)),
      ('relu4_4', nn.ReLU(inplace=True)),  # 28 x 28
      ('pad4_3',  nn.ReflectionPad2d(1)),
      ('conv4_3', nn.Conv2d(512, 512, 3, 1, 0)),
      ('relu4_3', nn.ReLU(inplace=True)),  # 28 x 28
      ('pad4_2',  nn.ReflectionPad2d(1)),
      ('conv4_2', nn.Conv2d(512, 512, 3, 1, 0)),
      ('relu4_2', nn.ReLU(inplace=True)),  # 28 x 28
      ('pad4_1',  nn.ReflectionPad2d(1)),
      ('conv4_1', nn.Conv2d(512, 256, 3, 1, 0)),
      ('relu4_1', nn.ReLU(inplace=True)),  # 28 x 28

      ('unpool3', nn.Upsample(scale_factor=2)),  # 28 x 28
      ('pad3_4',  nn.ReflectionPad2d(1)),
      ('conv3_4', nn.Conv2d(256, 256, 3, 1, 0)),
      ('relu3_4', nn.ReLU(inplace=True)),  # 56 x 56
      ('pad3_3',  nn.ReflectionPad2d(1)),
      ('conv3_3', nn.Conv2d(256, 256, 3, 1, 0)),
      ('relu3_3', nn.ReLU(inplace=True)),  # 56 x 56
      ('pad3_2',  nn.ReflectionPad2d(1)),
      ('conv3_2', nn.Conv2d(256, 256, 3, 1, 0)),
      ('relu3_2', nn.ReLU(inplace=True)),  # 56 x 56
      ('pad3_1',  nn.ReflectionPad2d(1)),
      ('conv3_1', nn.Conv2d(256, 128, 3, 1, 0)),
      ('relu3_1', nn.ReLU(inplace=True)),  # 56 x 56

      ('unpool2', nn.Upsample(scale_factor=2)),  # 56 x 56
      ('pad2_2',  nn.ReflectionPad2d(1)),
      ('conv2_2', nn.Conv2d(128, 128, 3, 1, 0)),
      ('relu2_2', nn.ReLU(inplace=True)),  # 112 x 112
      ('pad2_1',  nn.ReflectionPad2d(1)),# {{{}}}
      ('conv2_1', nn.Conv2d(128, 64, 3, 1, 0)),
      ('relu2_1', nn.ReLU(inplace=True)),  # 112 x 112

      ('unpool1', nn.Upsample(scale_factor=2)),  # 112 x 112
      ('pad1_2', nn.ReflectionPad2d(1)),
      ('conv1_2', nn.Conv2d(64, 64, 3, 1, 0)),
      ('relu1_2', nn.ReLU(inplace=True)),  # 224 x 224
      ('pad1_1', nn.ReflectionPad2d(1)),  # 226 x 226
      ('conv1_1', nn.Conv2d(64, 3, 3, 1, 0)),  # 224 x 224
    ])  # }}}

    self.seq = nn.Sequential(self.blocks)


  def forward(self, x, targets=None):
    return self.seq(x)


DECODERS = VGG19Decoder1, VGG19Decoder2, VGG19Decoder3, VGG19Decoder4, VGG19Decoder5
