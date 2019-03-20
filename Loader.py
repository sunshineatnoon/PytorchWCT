from PIL import Image
import skimage.transform
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.utils.data as data
from os import listdir
from os.path import join
import numpy as np
import torch
import os
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])

def default_loader(path):
    return Image.open(path).convert('RGB')

class Dataset(data.Dataset):
    def __init__(self,contentPath,stylePath,fineSize):
        super(Dataset,self).__init__()
        self.contentPath = contentPath
        self.image_list = [x for x in listdir(contentPath) if is_image_file(x)]
        self.stylePath = stylePath
        self.fineSize = fineSize
        self.prep = transforms.Compose([
                    Rescale(fineSize, mode='PIL'),
                    CropModulus(16),
                    transforms.ToTensor(),
                    ])

    def __getitem__(self,index):
        contentImgPath = os.path.join(self.contentPath,self.image_list[index])
        styleImgPath = os.path.join(self.stylePath,self.image_list[index])
        contentImg = default_loader(contentImgPath)
        styleImg = default_loader(styleImgPath)

        contentImg = self.prep(contentImg)
        styleImg = self.prep(styleImg)
        return contentImg, styleImg, self.image_list[index]

    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.image_list)


class CropModulus(object):

  def __init__(self, crop_modulus, mode='PIL'):
    assert mode in ['PIL', 'HWC', 'CHW']
    self.mode = mode
    self.crop_modulus = crop_modulus

  def __call__(self, im):
    if self.mode == 'PIL':
      W, H = im.size
    elif self.mode == 'HWC':
      H, W = im.shape[:2]
    elif self.mode == 'HWC':
      H, W = im.shape[1:3]
    Hmod = H - H % self.crop_modulus
    Wmod = W - W % self.crop_modulus
    border_x = (W - Wmod) // 2
    border_y = (H - Hmod) // 2
    end_x = border_x + Wmod
    end_y = border_y + Hmod
    crop_box = (border_x, border_y, end_x, end_y)
    if self.mode == 'PIL':
      return im.crop(crop_box)
    elif self.mode == 'HWC':
      return im[border_y:end_y, border_x:end_x, :]
    else: # self.mode == 'HWC':
      return im[: border_y:end_y, border_x:end_x]


class Rescale(object):
    """
    Rescale the image in a sample to a given size, preserving aspect ratio.
    If the input image's smaller dimension goes below the minimum size, scale
    *up* so that it matches.

    Args:
        target_size (int): Desired output size.
            The smaller (or bigger, if wished) of the image edges is matched
            to output_size keeping aspect ratio the same.
            If that puts the smaller edge below the minimum size,
            then the smaller of the edges is matched to output_size.
    """

    def __init__(self,
                 target_size,
                 max_size=2048, 
                 min_size=224,
                 scaling='smaller_side',
                 mode='numpy',
                 interpolation=Image.BILINEAR
                 ):
      assert mode in ['PIL', 'numpy', 'torch'], mode
      self.mode = mode
      self.interpolation = interpolation
      assert target_size >= 0, f'invalid target_size {target_size}'
      assert scaling in ['smaller_side', 'bigger_side']
      self.scaling = scaling
      assert min_size <= target_size, f'min_size = {min_size} <= target_size = {target_size}. Baaaad idea!'
      assert max_size <= 0 or target_size <= max_size, (target_size, max_size)
      self.min_size = min_size
      self.target_size = target_size
      self.max_size = max_size

    def target_shape(self, H, W, scaling=None):
      scaling = scaling or self.scaling
      if (scaling == 'bigger_side' and H > W) or (scaling == 'smaller_side' and H < W):
        Wnew = int(np.round(W/H * self.target_size))
        Hnew = self.target_size
      else:
        Wnew = self.target_size
        Hnew = int(np.round(H/W * self.target_size))
      return Hnew, Wnew

    def __call__(self, image):
      if self.mode == 'numpy':
        H, W = image.shape[:2]
      elif self.mode == 'torch':
        H, W = image.shape[1:3]
      else:  #self.mode == 'PIL':
        W, H = image.size
      scaling = self.scaling
      target_size = self.target_size

      Hnew, Wnew = self.target_shape(H, W)

      if Wnew < self.min_size or Hnew < self.min_size:
        # this can only happen with scaling=='bigger_side'
        # print(f'WARNING: image is too small after scaling. scaling UP instead of down.')
        Hnew, Wnew = self.target_shape(H, W, 'smaller_side')

      if self.mode == 'numpy':
        return skimage.transform.resize(image, (Hnew, Wnew), preserve_range=True)
      elif self.mode == 'torch':
        # apparently, there is no simple way to resize a tensor 0_o
        image = tvt.functional.to_pil_image(image, mode='RGB')
        image = image.resize((Wnew, Hnew), resample=self.interpolation)
        image = tvt.functional.to_tensor(image)
        return image
      else:  # self.mode in ['PIL', 'torch']:
        return image.resize((Wnew, Hnew), resample=self.interpolation)


