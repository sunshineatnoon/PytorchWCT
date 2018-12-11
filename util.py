from __future__ import division
import torch
import torchvision.transforms as transforms
import numpy as np
import argparse
import time
import os
from PIL import Image
from vgg19_decoders import VGG19Decoder1, VGG19Decoder2, VGG19Decoder3, VGG19Decoder4, VGG19Decoder5 
from vgg19_normalized import VGG19_normalized
import torch.nn as nn



class WCT(nn.Module):
    def __init__(self,args):
        super(WCT, self).__init__()
        # load pre-trained network
        self.encoder = VGG19_normalized()
        self.encoder.load_state_dict(torch.load(args.encoder))

        self.d1 = VGG19Decoder1()
        self.d1.load_state_dict(torch.load(args.decoder1))
        self.d2 = VGG19Decoder2()
        self.d2.load_state_dict(torch.load(args.decoder2))
        self.d3 = VGG19Decoder3()
        self.d3.load_state_dict(torch.load(args.decoder3))
        self.d4 = VGG19Decoder4()
        self.d4.load_state_dict(torch.load(args.decoder4))
        self.d5 = VGG19Decoder5()
        self.d5.load_state_dict(torch.load(args.decoder5))

    def whiten_and_color(self,cF,sF):
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

    def transform(self,cF,sF,alpha):
        cF = cF.double()
        sF = sF.double()
        C,W,H = cF.size(0),cF.size(1),cF.size(2)
        _,W1,H1 = sF.size(0),sF.size(1),sF.size(2)
        cFView = cF.view(C,-1)
        sFView = sF.view(C,-1)

        targetFeature = self.whiten_and_color(cFView,sFView)
        targetFeature = targetFeature.view_as(cF)
        csF = alpha * targetFeature + (1.0 - alpha) * cF
        csF = csF.float().unsqueeze(0)
        return csF
