#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 17:08:32 2017

@author: a
"""

import sys
sys.path.append("/home/a/caffe/python")
import caffe
import numpy as np
import os
import sys

#weights = '../ilsvrc-nets/vgg16-fcn.caffemodel'
# init
#caffe.set_device(int(sys.argv[1]))

caffe.set_mode_cpu()

solver = caffe.SGDSolver('srcnn_sovler.prototxt')
#solver.net.copy_from(weights)

for _ in range(25):
    solver.step(100)

