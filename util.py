from __future__ import division
import torch
import numpy as np
import argparse
import time
import os
from PIL import Image
from vgg19_decoders import VGG19Decoder1, VGG19Decoder2, VGG19Decoder3, VGG19Decoder4, VGG19Decoder5 
from vgg19_normalized import VGG19_normalized
import torch.nn as nn



def matrix_sqrt(A):
  A = A.clone()
  a_diag_ = A.diagonal()
  a_diag_ += 1e-4

  s_u, s_e, s_v = torch.svd(A,some=False)

  k_s = A.shape[-1]
  for i in range(k_s):
      if s_e[i] < 0.00001:
          k_s = i
          break

  s_d = (s_e[0:k_s]).pow(0.5)
  step1 = torch.mm(s_v[:,0:k_s], torch.diag(s_d))
  result = torch.mm(step1, (s_v[:,0:k_s].t()))
  return result


def matrix_inv_sqrt(A):
  A = A.clone()
  a_diag_ = A.diagonal()
  a_diag_ += 1e-4
  k_c = A.shape[-1]
  c_u,c_e,c_v = torch.svd(A, some=False)

  for i in range(k_c):
      if c_e[i] < 0.00001:
          k_c = i
          break

  c_d = (c_e[0:k_c]).pow(-0.5)
  step1 = torch.mm(c_v[:,0:k_c],torch.diag(c_d))
  result = torch.mm(step1,(c_v[:,0:k_c].t()))
  return result


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

        self.decoders = {'relu1_1': self.d1,
                         'relu2_1': self.d2,
                         'relu3_1': self.d3,
                         'relu4_1': self.d4,
                         'relu5_1': self.d5}

    def whiten_and_color(self,cF,sF, method):
        cFSize = cF.size()
        print(f'cF.shape = {cF.shape}')
        c_mean = torch.mean(cF,1) # c x (h x w)
        c_mean = c_mean.unsqueeze(1).expand_as(cF)
        cF = cF - c_mean

        contentConv = torch.mm(cF,cF.t()).div(cFSize[1]-1) + torch.eye(cFSize[0]).double()

        sFSize = sF.size()
        s_mean = torch.mean(sF,1)
        sF = sF - s_mean.unsqueeze(1).expand_as(sF)
        styleConv = torch.mm(sF,sF.t()).div(sFSize[1]-1)

        if method == 'original':  # the original WCT by Li et al.
          cF_inv_sqrt = matrix_inv_sqrt(contentConv)
          sF_sqrt = matrix_sqrt(styleConv)
          # whiten_cF = torch.mm(cF_inv_sqrt, cF)
          # targetFeature = torch.mm(sF_sqrt,whiten_cF)
          targetFeature = sF_sqrt @ (cF_inv_sqrt @ cF)
        else:  # Lu et al.
          assert method == 'closed-form'
          cF_sqrt = matrix_sqrt(contentConv)
          cF_inv_sqrt = matrix_inv_sqrt(contentConv)
          print(f'cF_sqrt.shape = {cF_sqrt.shape}')
          middle_matrix = matrix_sqrt(cF_sqrt @ styleConv @ cF_sqrt)
          print(f'middle_matrix.shape = {middle_matrix.shape}')
          transform_matrix = cF_inv_sqrt @ middle_matrix @ cF_inv_sqrt
          targetFeature = transform_matrix @ cF
          print(f'targetFeature.shape = {targetFeature.shape}')

        targetFeature = targetFeature + s_mean.unsqueeze(1).expand_as(targetFeature)
        return targetFeature

    def transform(self, cF, sF, method):
        cF = cF.double()
        sF = sF.double()
        C,W,H = cF.size(0),cF.size(1),cF.size(2)
        _,W1,H1 = sF.size(0),sF.size(1),sF.size(2)
        cFView = cF.view(C,-1)
        sFView = sF.view(C,-1)

        targetFeature = self.whiten_and_color(cFView, sFView, method)
        targetFeature = targetFeature.view_as(cF)
        return targetFeature

