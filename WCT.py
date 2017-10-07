import os
import torch
import argparse
from PIL import Image
from torch.autograd import Variable
import torchvision.utils as vutils
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from Loader import Dataset
from util import *
import scipy.misc
from torch.utils.serialization import load_lua
import time

parser = argparse.ArgumentParser(description='WCT Pytorch')
parser.add_argument('--contentPath',default='images/content',help='path to train')
parser.add_argument('--stylePath',default='images/style',help='path to train')
parser.add_argument('--workers', default=2, type=int, metavar='N',help='number of data loading workers (default: 4)')
parser.add_argument('--vgg1', default='/home/xtli/WEIGHTS/WCT_Pytorch/vgg_normalised_conv1_1.t7', help='Path to the VGG conv1_1')
parser.add_argument('--vgg2', default='/home/xtli/WEIGHTS/WCT_Pytorch/vgg_normalised_conv2_1.t7', help='Path to the VGG conv2_1')
parser.add_argument('--vgg3', default='/home/xtli/WEIGHTS/WCT_Pytorch/vgg_normalised_conv3_1.t7', help='Path to the VGG conv3_1')
parser.add_argument('--vgg4', default='/home/xtli/WEIGHTS/WCT_Pytorch/vgg_normalised_conv4_1.t7', help='Path to the VGG conv4_1')
parser.add_argument('--vgg5', default='/home/xtli/WEIGHTS/WCT_Pytorch/vgg_normalised_conv5_1.t7', help='Path to the VGG conv5_1')
parser.add_argument('--decoder5', default='/home/xtli/WEIGHTS/WCT_Pytorch/feature_invertor_conv5_1.t7', help='Path to the decoder5')
parser.add_argument('--decoder4', default='/home/xtli/WEIGHTS/WCT_Pytorch/feature_invertor_conv4_1.t7', help='Path to the decoder4')
parser.add_argument('--decoder3', default='/home/xtli/WEIGHTS/WCT_Pytorch/feature_invertor_conv3_1.t7', help='Path to the decoder3')
parser.add_argument('--decoder2', default='/home/xtli/WEIGHTS/WCT_Pytorch/feature_invertor_conv2_1.t7', help='Path to the decoder2')
parser.add_argument('--decoder1', default='/home/xtli/WEIGHTS/WCT_Pytorch/feature_invertor_conv1_1.t7', help='Path to the decoder1')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--fineSize', type=int, default=512, help='resize image to fineSize x fineSize,leave it to 0 if not resize')
parser.add_argument('--outf', default='samples/', help='folder to output images')
parser.add_argument('--alpha', type=float,default=1, help='hyperparameter to blend wct feature and content feature')
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
def scale_dialate(seg,W,H):
    # TODO: dialate
    seg = seg.unsqueeze(0)
    seg = seg.expand(3,seg.size(1),seg.size(2))
    seg = transforms.ToPILImage()(seg)
    seg = seg.resize((H,W))
    seg = transforms.ToTensor()(seg)[0].squeeze(0)
    return seg

def wct2(cF,sF):
    cFSize = cF.size()
    c_mean = torch.mean(cF,1) # c x (h x w)
    c_mean = c_mean.unsqueeze(1).expand_as(cF)
    cF = cF - c_mean

    contentConv = torch.mm(cF,cF.t()).div(cFSize[1]-1) + torch.eye(cFSize[0]).double()
    c_u,c_e,c_v = torch.svd(contentConv,some=False)

    k_c = cFSize[0]
    for i in range(cFSize[0]):
        if c_e[i] < 0.00001:
            k_c = i
            break

    sFSize = sF.size()
    s_mean = torch.mean(sF,1)
    sF = sF - s_mean.unsqueeze(1).expand_as(sF)
    styleConv = torch.mm(sF,sF.t()).div(sFSize[1]-1)
    s_u,s_e,s_v = torch.svd(styleConv,some=False)

    k_s = sFSize[0]
    for i in range(sFSize[0]):
        if s_e[i] < 0.00001:
            k_s = i
            break

    c_d = (c_e[0:k_c]).pow(-0.5)
    step1 = torch.mm(c_v[:,0:k_c],torch.diag(c_d))
    step2 = torch.mm(step1,(c_v[:,0:k_c].t()))
    whiten_cF = torch.mm(step2,cF)

    s_d = (s_e[0:k_s]).pow(0.5)
    targetFeature = torch.mm(torch.mm(torch.mm(s_v[:,0:k_s],torch.diag(s_d)),(s_v[:,0:k_s].t())),whiten_cF)
    targetFeature = targetFeature + s_mean.unsqueeze(1).expand_as(targetFeature)
    return targetFeature

def feature_wct(cF,sF,csF):
    cF = cF.double()
    sF = sF.double()
    C,W,H = cF.size(0),cF.size(1),cF.size(2)
    _,W1,H1 = sF.size(0),sF.size(1),sF.size(2)
    cFView = cF.view(C,-1)
    sFView = sF.view(C,-1)

    targetFeature = wct2(cFView,sFView)
    targetFeature = targetFeature.view_as(cF)
    ccsF = args.alpha * targetFeature + (1.0 - args.alpha) * cF
    ccsF = ccsF.float().unsqueeze(0)
    csF.data.resize_(ccsF.size()).copy_(ccsF)
    return csF

def styleTransfer(contentImg,styleImg,imname,csF):
    e1,d1,e2,d2,e3,d3,e4,d4,e5,d5 = loadModel(args)

    sF5 = e5(styleImg)
    cF5 = e5(contentImg)
    sF5 = sF5.data.cpu().squeeze(0)
    cF5 = cF5.data.cpu().squeeze(0)
    csF5 = feature_wct(cF5,sF5,csF)
    Im5 = d5(csF5)

    sF4 = e4(Im5)
    cF4 = e4(contentImg)
    sF4 = sF4.data.cpu().squeeze(0)
    cF4 = cF4.data.cpu().squeeze(0)
    csF4 = feature_wct(cF4,sF4,csF)
    Im4 = d4(csF4)

    sF3 = e3(styleImg)
    cF3 = e3(Im4)
    sF3 = sF3.data.cpu().squeeze(0)
    cF3 = cF3.data.cpu().squeeze(0)
    csF3 = feature_wct(cF3,sF3,csF)
    Im3 = d3(csF3)

    sF2= e2(styleImg)
    cF2 = e2(Im3)
    sF2 = sF2.data.cpu().squeeze(0)
    cF2 = cF2.data.cpu().squeeze(0)
    csF2 = feature_wct(cF2,sF2,csF)
    Im2 = d2(csF2)

    sF1 = e1(styleImg)
    cF1 = e1(Im2)
    sF1 = sF1.data.cpu().squeeze(0)
    cF1 = cF1.data.cpu().squeeze(0)
    csF1 = feature_wct(cF1,sF1,csF)
    Im1 = d1(csF1)
    # save_image has this wired design to pad images with 4 pixels at default.
    vutils.save_image(Im1.data.cpu().float(),os.path.join(args.outf,imname))
    return

avgTime = 0
cImg = torch.Tensor()
sImg = torch.Tensor()
csF = torch.Tensor()
cImg = Variable(cImg)
sImg = Variable(sImg)
csF = Variable(csF)
if(args.cuda):
    cImg = cImg.cuda()
    sImg = sImg.cuda()
    csF = csF.cuda()
for i,(contentImg,styleImg,imname) in enumerate(loader):
    imname = imname[0]
    print('Transferring ' + imname)
    cImg.data.resize_(contentImg.size()).copy_(contentImg)
    sImg.data.resize_(styleImg.size()).copy_(styleImg)
    start_time = time.time()
    # WCT Style Transfer
    styleTransfer(cImg,sImg,imname,csF)
    end_time = time.time()
    print('Elapsed time is: %f' % (end_time - start_time))
    avgTime += (end_time - start_time)

print('Processed %d images. Averaged time is %f' % ((i+1),avgTime/(i+1)))
