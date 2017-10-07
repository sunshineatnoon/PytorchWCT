from __future__ import division
import torch
from torch.utils.serialization import load_lua
import numpy as np
import argparse
import time
import os
from PIL import Image
from modelsNIPS import decoder1,decoder2,decoder3,decoder4,decoder5
from modelsNIPS import encoder1,encoder2,encoder3,encoder4,encoder5


def loadModel(args):

    # load pre-trained network
    vgg1 = load_lua(args.vgg1)
    decoder1_torch = load_lua(args.decoder1)
    vgg2 = load_lua(args.vgg2)
    decoder2_torch = load_lua(args.decoder2)
    vgg3 = load_lua(args.vgg3)
    decoder3_torch = load_lua(args.decoder3)
    vgg4 = load_lua(args.vgg4)
    decoder4_torch = load_lua(args.decoder4)
    vgg5 = load_lua(args.vgg5)
    decoder5_torch = load_lua(args.decoder5)


    e1 = encoder1(vgg1)
    d1 = decoder1(decoder1_torch)
    e2 = encoder2(vgg2)
    d2 = decoder2(decoder2_torch)
    e3 = encoder3(vgg3)
    d3 = decoder3(decoder3_torch)
    e4 = encoder4(vgg4)
    d4 = decoder4(decoder4_torch)
    e5 = encoder5(vgg5)
    d5 = decoder5(decoder5_torch)
    if(args.cuda):
        e1.cuda()
        e2.cuda()
        e3.cuda()
        e4.cuda()
        e5.cuda()
        d1.cuda()
        d2.cuda()
        d3.cuda()
        d4.cuda()
        d5.cuda()

    # save some space
    del vgg1; del decoder1_torch
    del vgg2; del decoder2_torch
    del vgg3; del decoder3_torch
    del vgg4; del decoder4_torch
    del vgg5; del decoder5_torch
    return e1,d1,e2,d2,e3,d3,e4,d4,e5,d5
