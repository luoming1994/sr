#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 17:08:32 2017

@author: a
"""

import sys
sys.path.append("/home/a/caffe/python")
import caffe
from caffe import layers as L, params as P
from caffe.coord_map import crop


def conv_relu(bottom, nout, ks=3, stride=1, pad=1):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
        num_output=nout, pad=pad,
        #weight_filler=dict(type='xavier'),
	weight_filler=dict(type='gaussian',std=0.001),
        #param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
	param=[dict(lr_mult=1), dict(lr_mult=0.1)])
    return conv, L.ReLU(conv, in_place=True)

def max_pool(bottom, ks=2, stride=2):
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)

def init_SRCNN(string):
    n = caffe.NetSpec()
    # date + label
    if string=='train':
        n.data = L.Data(source='/home/a/SR/trainL_lmdb', backend=P.Data.LMDB, 
                        batch_size=1, ntop=1,include=dict(phase=caffe.TRAIN))
        n.label= L.Data(source='/home/a/SR/trainH_lmdb', backend=P.Data.LMDB, 
                       batch_size=1, ntop=1,include=dict(phase=caffe.TRAIN))
    else:
        n.data = L.Data(source='/home/a/SR/testL_lmdb', backend=P.Data.LMDB, 
                        batch_size=1, ntop=1,include=dict(phase=caffe.TEST))
        n.label= L.Data(source='/home/a/SR/testH_lmdb', backend=P.Data.LMDB, 
                        batch_size=1, ntop=1,include=dict(phase=caffe.TEST))
    # the base net 
    n.conv1, n.relu1 = conv_relu(n.data,64, ks=9,pad=4)
    n.conv2, n.relu2 = conv_relu(n.relu1, 32,ks=1,pad=0)
    n.conv3, n.relu3 = conv_relu(n.relu2, 3,ks=5,pad=2)
   
    n.loss = L.EuclideanLoss(n.conv3, n.label)
    #n.acc = L.Accuracy(n.score, n.label)
    
    return n.to_proto()

def make_net():
    with open('srcnn_train.prototxt', 'w') as f:
        f.write(str(init_SRCNN('train')))
    with open('srcnn_test.prototxt', 'w') as f:
        f.write(str(init_SRCNN('test')))


if __name__ == '__main__':
    make_net()
