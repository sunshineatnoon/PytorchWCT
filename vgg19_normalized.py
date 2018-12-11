import torch
import torch.nn as nn
from collections import OrderedDict

class VGG19_normalized(nn.Module):
  def __init__(self):
    """
    VGG19 normalized.
    Takes RGB within [0, 1] as input.
    Do NOT normalize the data as with other VGG models!
    """

    super(VGG19_normalized,self).__init__()

    #self.preprocess_weight =
    self.register_buffer(
      'preprocess_weight',
      torch.FloatTensor([[[[  0.]], [[  0.]], [[255.]]],
                         [[[  0.]], [[255.]], [[  0.]]],
                         [[[255.]], [[  0.]], [[  0.]]]]))
    #self.preprocess_bias =
    self.register_buffer(
      'preprocess_bias',
      torch.FloatTensor([-103.9390, -116.7790, -123.6800]))

    # input shape originally 224 x 224

    self.blocks = OrderedDict([
      ('pad1_1', nn.ReflectionPad2d(1)),  # 226 x 226
      ('conv1_1', nn.Conv2d(3, 64, 3, 1, 0)),
      ('relu1_1', nn.ReLU(inplace=True)),  # 224 x 224
      ('pad1_2', nn.ReflectionPad2d(1)),
      ('conv1_2', nn.Conv2d(64, 64, 3, 1, 0)),
      ('relu1_2', nn.ReLU(inplace=True)),  # 224 x 224
      ('pool1', nn.MaxPool2d(kernel_size=2, stride=2)),  # 112 x 112

      ('pad2_1',  nn.ReflectionPad2d(1)),# {{{}}}
      ('conv2_1', nn.Conv2d(64, 128, 3, 1, 0)),
      ('relu2_1', nn.ReLU(inplace=True)),  # 112 x 112
      ('pad2_2',  nn.ReflectionPad2d(1)),
      ('conv2_2', nn.Conv2d(128, 128, 3, 1, 0)),
      ('relu2_2', nn.ReLU(inplace=True)),  # 112 x 112
      ('pool2', nn.MaxPool2d(kernel_size=2, stride=2)),  # 56 x 56

      ('pad3_1',  nn.ReflectionPad2d(1)),
      ('conv3_1', nn.Conv2d(128, 256, 3, 1, 0)),
      ('relu3_1', nn.ReLU(inplace=True)),  # 56 x 56
      ('pad3_2',  nn.ReflectionPad2d(1)),
      ('conv3_2', nn.Conv2d(256, 256, 3, 1, 0)),
      ('relu3_2', nn.ReLU(inplace=True)),  # 56 x 56
      ('pad3_3',  nn.ReflectionPad2d(1)),
      ('conv3_3', nn.Conv2d(256, 256, 3, 1, 0)),
      ('relu3_3', nn.ReLU(inplace=True)),  # 56 x 56
      ('pad3_4',  nn.ReflectionPad2d(1)),
      ('conv3_4', nn.Conv2d(256, 256, 3, 1, 0)),
      ('relu3_4', nn.ReLU(inplace=True)),  # 56 x 56
      ('pool3', nn.MaxPool2d(kernel_size=2, stride=2)),  # 28 x 28

      ('pad4_1',  nn.ReflectionPad2d(1)),
      ('conv4_1', nn.Conv2d(256, 512, 3, 1, 0)),
      ('relu4_1', nn.ReLU(inplace=True)),  # 28 x 28
      ('pad4_2',  nn.ReflectionPad2d(1)),
      ('conv4_2', nn.Conv2d(512, 512, 3, 1, 0)),
      ('relu4_2', nn.ReLU(inplace=True)),  # 28 x 28
      ('pad4_3',  nn.ReflectionPad2d(1)),
      ('conv4_3', nn.Conv2d(512, 512, 3, 1, 0)),
      ('relu4_3', nn.ReLU(inplace=True)),  # 28 x 28
      ('pad4_4',  nn.ReflectionPad2d(1)),
      ('conv4_4', nn.Conv2d(512, 512, 3, 1, 0)),
      ('relu4_4', nn.ReLU(inplace=True)),  # 28 x 28
      ('pool4', nn.MaxPool2d(kernel_size=2, stride=2)),  # 14 x 14

      ('pad5_1',  nn.ReflectionPad2d(1)),
      ('conv5_1', nn.Conv2d(512, 512, 3, 1, 0)),
      ('relu5_1', nn.ReLU(inplace=True)),  # 14 x 14
    ])

    self.seq = nn.Sequential(self.blocks)


  def forward(self, x, targets=None):
    # don't want this one to be trainable so we don't make it into parameters
    out = nn.functional.conv2d(x,
                               weight=self.preprocess_weight,
                               bias=self.preprocess_bias)

    # by default, just run the whole thing
    targets = targets or 'relu5_1'

    if isinstance(targets, str):
      assert targets in self.blocks.keys(), f'"{targets}" is not a valid target'
      for n, b in self.blocks.items():
        out = b(out)
        if n == targets:
          return out
    

    for t in targets:
      assert t in self.blocks.keys(), f'"{t}" is not a valid target'
    
    results = dict()
    for n, b in self.blocks.items():
      out = b(out)
      if n in targets:
        results[n] == out
      if len(results) == len(set(targets)):
        break

    results = [results[t] for t in targets]
    return results
