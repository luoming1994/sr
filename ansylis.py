#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 15:15:21 2017

@author: a
"""
import numpy as np
from PIL import Image

def load_img(img_path):
    #in Channel x Height x Width order
    im = np.array(Image.open(img_path), dtype=np.float32) # or load whatever ndarray you need
    im = im[:,:,::-1]
    im = im.transpose((2,0,1))
    return im

def PSNR(im_path,gt_path):
    im = load_img(im_path)
    gt = load_img(gt_path)
    im_shape = im.shape
    gt_shape = gt.shape
    if gt_shape != im_shape:
        return -1
    mse = np.mean((gt - im)**2)
    psnr = 10*np.log10(255**2/mse)
    return psnr,mse

def SSIM(im_path,gt_path):
    im = load_img(im_path)
    gt = load_img(gt_path)
    im_shape = im.shape
    gt_shape = gt.shape
    if gt_shape != im_shape:
        return -1   
    
    # C1=(K1*L)^2, 
    # C2=(K2*L)^2
    # C3=C2/2,     1=0.01, K2=0.03, L=255
    C1 = (0.01*255)**2
    C2 = (0.03*255)**2
    C3 = C2/2.0
    
    mean_x = im.mean() # mean of im
    mean_y = gt.mean() # mean of gt
    cov = np.cov([gt.flatten(),im.flatten()])
    cov_xx = cov[0,0]
    cov_x = np.sqrt(cov_xx)
    cov_yy= cov[1,1]
    cov_y = np.sqrt(cov_yy) 
    cov_xy = cov[0,1]
    
    l_xy = (2*mean_x*mean_y + C1) / (mean_x**2 + mean_y**2 + C1)
    c_xy = (2*cov_x*cov_y + C2) / (cov_xx + cov_yy + C2)
    s_xy = (cov_xy + C3) / (cov_x*cov_y + C3)
    ssim = l_xy*c_xy*s_xy
    
    return ssim

if __name__ == '__main__':
    gt_path = r'/home/a/SR/t1.bmp'
    im_path = r'/home/a/SR/1.bmp'
    psnr,mse = PSNR(im_path,gt_path)
    ssim = SSIM(im_path,gt_path)
