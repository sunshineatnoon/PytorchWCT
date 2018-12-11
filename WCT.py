import os
import torch
import argparse
from PIL import Image
from torch.autograd import Variable
import torchvision.utils as vutils
import torchvision.datasets as datasets
from Loader import Dataset
from util import *
import scipy.misc
import time

parser = argparse.ArgumentParser(description='WCT Pytorch')
parser.add_argument('--contentPath',default='images/content',help='path to train')
parser.add_argument('--stylePath',default='images/style',help='path to train')
parser.add_argument('--workers', default=2, type=int, metavar='N',help='number of data loading workers (default: 4)')
parser.add_argument('--encoder', default='models/vgg19_normalized.pth.tar', help='Path to the VGG conv1_1')
parser.add_argument('--decoder5', default='models/vgg19_normalized_decoder5.pth.tar', help='Path to the decoder5')
parser.add_argument('--decoder4', default='models/vgg19_normalized_decoder4.pth.tar', help='Path to the decoder4')
parser.add_argument('--decoder3', default='models/vgg19_normalized_decoder3.pth.tar', help='Path to the decoder3')
parser.add_argument('--decoder2', default='models/vgg19_normalized_decoder2.pth.tar', help='Path to the decoder2')
parser.add_argument('--decoder1', default='models/vgg19_normalized_decoder1.pth.tar', help='Path to the decoder1')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--fineSize', type=int, default=512, help='resize image to fineSize x fineSize,leave it to 0 if not resize')
parser.add_argument('--outf', default='samples/', help='folder to output images')
parser.add_argument('--alpha', type=float,default=1, help='hyperparameter to blend wct feature and content feature')
parser.add_argument('--gpu', type=int, default=0, help="which gpu to run on.  default is 0")

args = parser.parse_args()

try:
    os.makedirs(args.outf)
except OSError:
    pass

# Data loading code
dataset = Dataset(args.contentPath,args.stylePath,args.fineSize)
loader = torch.utils.data.DataLoader(dataset=dataset,
                                     batch_size=1,
                                     shuffle=False)

def styleTransfer(wct, contentImg, styleImg, imname):

    sF5 = wct.encoder(styleImg, 'relu5_1')
    cF5 = wct.encoder(contentImg, 'relu5_1')
    sF5 = sF5.cpu().squeeze(0)
    cF5 = cF5.cpu().squeeze(0)
    csF5 = wct.transform(cF5,sF5,args.alpha)
    csF5 = csF5.to(next(wct.parameters()).device)
    Im5 = wct.d5(csF5)

    sF4 = wct.encoder(styleImg, 'relu4_1')
    cF4 = wct.encoder(Im5, 'relu4_1')
    sF4 = sF4.cpu().squeeze(0)
    cF4 = cF4.cpu().squeeze(0)
    csF4 = wct.transform(cF4,sF4,args.alpha)
    csF4 = csF4.to(next(wct.parameters()).device)
    Im4 = wct.d4(csF4)

    sF3 = wct.encoder(styleImg, 'relu3_1')
    cF3 = wct.encoder(Im4, 'relu3_1')
    sF3 = sF3.cpu().squeeze(0)
    cF3 = cF3.cpu().squeeze(0)
    csF3 = wct.transform(cF3,sF3,args.alpha)
    csF3 = csF3.to(next(wct.parameters()).device)
    Im3 = wct.d3(csF3)

    sF2 = wct.encoder(styleImg, 'relu2_1')
    cF2 = wct.encoder(Im3, 'relu2_1')
    sF2 = sF2.cpu().squeeze(0)
    cF2 = cF2.cpu().squeeze(0)
    csF2 = wct.transform(cF2,sF2,args.alpha)
    csF2 = csF2.to(next(wct.parameters()).device)
    Im2 = wct.d2(csF2)

    sF1 = wct.encoder(styleImg, 'relu1_1')
    cF1 = wct.encoder(Im2, 'relu1_1')
    sF1 = sF1.cpu().squeeze(0)
    cF1 = cF1.cpu().squeeze(0)
    csF1 = wct.transform(cF1,sF1,args.alpha)
    csF1 = csF1.to(next(wct.parameters()).device)
    Im1 = wct.d1(csF1)
    # save_image has this wired design to pad images with 4 pixels at default.
    vutils.save_image(Im1.cpu().float(),os.path.join(args.outf,imname))
    return

def main():
  wct = WCT(args)
  if(args.cuda):
      wct.cuda(args.gpu)

  avgTime = 0
  for i,(contentImg,styleImg,imname) in enumerate(loader):
      if(args.cuda):
          contentImg = contentImg.cuda(args.gpu)
          styleImg = styleImg.cuda(args.gpu)
      imname = imname[0]
      print('Transferring ' + imname)
      if (args.cuda):
          contentImg = contentImg.cuda(args.gpu)
          styleImg = styleImg.cuda(args.gpu)
      start_time = time.time()
      # WCT Style Transfer
      styleTransfer(wct, contentImg, styleImg, imname)
      end_time = time.time()
      print('Elapsed time is: %f' % (end_time - start_time))
      avgTime += (end_time - start_time)

  print('Processed %d images. Averaged time is %f' % ((i+1),avgTime/(i+1)))

if __name__ == '__main__':
  with torch.no_grad():
    main()
