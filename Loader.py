from PIL import Image
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

def MaskHelper(seg,color):
    # green
    mask = torch.Tensor()
    if(color == 'green'):
        mask = torch.lt(seg[0],0.1)
        mask = torch.mul(mask,torch.gt(seg[1],1-0.1))
        mask = torch.mul(mask,torch.lt(seg[2],0.1))
    elif(color == 'black'):
        mask = torch.lt(seg[0], 0.1)
        mask = torch.mul(mask,torch.lt(seg[1], 0.1))
        mask = torch.mul(mask,torch.lt(seg[2], 0.1))
    elif(color == 'white'):
        mask = torch.gt(seg[0], 1-0.1)
        mask = torch.mul(mask,torch.gt(seg[1], 1-0.1))
        mask = torch.mul(mask,torch.gt(seg[2], 1-0.1))
    elif(color == 'red'):
        mask = torch.gt(seg[0], 1-0.1)
        mask = torch.mul(mask,torch.lt(seg[1], 0.1))
        mask = torch.mul(mask,torch.lt(seg[2], 0.1))
    elif(color == 'blue'):
        mask = torch.lt(seg[0], 0.1)
        mask = torch.mul(mask,torch.lt(seg[1], 0.1))
        mask = torch.mul(mask,torch.gt(seg[2], 1-0.1))
    elif(color == 'yellow'):
        mask = torch.gt(seg[0], 1-0.1)
        mask = torch.mul(mask,torch.gt(seg[1], 1-0.1))
        mask = torch.mul(mask,torch.lt(seg[2], 0.1))
    elif(color == 'grey'):
        mask = torch.lt(seg[0], 0.1)
        mask = torch.mul(mask,torch.lt(seg[1], 0.1))
        mask = torch.mul(mask,torch.lt(seg[2], 0.1))
    elif(color == 'lightblue'):
        mask = torch.lt(seg[0], 0.1)
        mask = torch.mul(mask,torch.gt(seg[1], 1-0.1))
        mask = torch.mul(mask,torch.gt(seg[2], 1-0.1))
    elif(color == 'purple'):
        mask = torch.gt(seg[0], 1-0.1)
        mask = torch.mul(mask,torch.lt(seg[1], 0.1))
        mask = torch.mul(mask,torch.gt(seg[2], 1-0.1))
    else:
        print('MaskHelper(): color not recognized, color = ' + color)
    return mask.float()

def ExtractMask(contentSeg,styleSeg):
    # Given segmentation for content and style, we get a list of segmentation for each color
    '''
    Test Code:
        content_masks,style_masks = ExtractMask(contentSegImg,styleSegImg)
        for i,mask in enumerate(content_masks):
            vutils.save_image(mask,'samples/content_%d.png' % (i),normalize=True)
        for i,mask in enumerate(style_masks):
            vutils.save_image(mask,'samples/style_%d.png' % (i),normalize=True)
    '''
    color_codes = ['blue', 'green', 'black', 'white', 'red', 'yellow', 'grey', 'lightblue', 'purple']
    content_masks = []
    style_masks = []
    for color in color_codes:
        content_mask = MaskHelper(contentSeg,color)
        style_mask = MaskHelper(styleSeg,color)
        content_masks.append(content_mask)
        style_masks.append(style_mask)
    return content_masks,style_masks


class Dataset(data.Dataset):
    def __init__(self,contentPath,stylePath,contentSegPath,styleSegPath,fineSize):
        super(Dataset,self).__init__()
        self.contentPath = contentPath
        self.image_list = [x for x in listdir(contentPath) if is_image_file(x)]
        self.stylePath = stylePath
        self.contentSegPath = contentSegPath
        self.styleSegPath = styleSegPath
        self.fineSize = fineSize
        #self.normalize = transforms.Normalize(mean=[103.939,116.779,123.68],std=[1, 1, 1])
        #normalize = transforms.Normalize(mean=[123.68,103.939,116.779],std=[1, 1, 1])
        self.prep = transforms.Compose([
                    transforms.Scale(fineSize),
                    transforms.ToTensor(),
                    #transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]), #turn to BGR
                    ])

    def __getitem__(self,index):
        contentImgPath = os.path.join(self.contentPath,self.image_list[index])
        styleImgPath = os.path.join(self.stylePath,self.image_list[index])
        contentImg = default_loader(contentImgPath)
        styleImg = default_loader(styleImgPath)
        try:
            contentSegImgPath = os.path.join(self.contentSegPath,self.image_list[index])
            styleSegImgPath = os.path.join(self.styleSegPath,self.image_list[index])
            contentSegImg = default_loader(contentSegImgPath)
            styleSegImg = default_loader(styleSegImgPath)
        except:
            # if the user doesn't give as mask, we fake one with zeros
            contentSegImg = Image.new('RGB', (contentImg.size))
            styleSegImg = Image.new('RGB', (styleImg.size))


        # resize
        if(self.fineSize != 0):
            w,h = contentImg.size
            if(w > h):
                if(w != self.fineSize):
                    neww = self.fineSize
                    newh = h*neww/w
                    contentImg = contentImg.resize((neww,newh))
                    styleImg = styleImg.resize((neww,newh))
                    contentSegImg = contentSegImg.resize((neww,newh))
                    styleSegImg = styleSegImg.resize((neww,newh))
            else:
                if(h != self.fineSize):
                    newh = self.fineSize
                    neww = w*newh/h
                    contentImg = contentImg.resize((neww,newh))
                    styleImg = styleImg.resize((neww,newh))
                    contentSegImg = contentSegImg.resize((neww,newh))
                    styleSegImg = styleSegImg.resize((neww,newh))


        # Turning segmentation images into masks
        styleSegImg = transforms.ToTensor()(styleSegImg)
        contentSegImg = transforms.ToTensor()(contentSegImg)
        content_masks,style_masks = ExtractMask(contentSegImg,styleSegImg)

        # Preprocess Images
        contentImg = transforms.ToTensor()(contentImg)
        styleImg = transforms.ToTensor()(styleImg)
        return contentImg.squeeze(0),styleImg.squeeze(0),content_masks,style_masks,self.image_list[index]

    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.image_list)
