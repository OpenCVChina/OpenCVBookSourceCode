#!/usr/bin/env python3
# encoding:utf-8


import cv2 as cv
import numpy as np


def main():

    # 读入图像
    im = cv.imread('lena.jpg')
    cv.imshow('lena.jpg', im)

    # 缩放图像
    dim = (int(im.shape[1]*1.3), int(im.shape[0]*1.3))
    im_rs_nr = cv.resize(im, dim, interpolation=cv.INTER_NEAREST)
    im_rs_ln = cv.resize(im, dim, interpolation=cv.INTER_LINEAR)
    im_rs_cb = cv.resize(im, dim, interpolation=cv.INTER_CUBIC)
    im_rs_lz = cv.resize(im, dim, interpolation=cv.INTER_LANCZOS4)
    cv.imshow('lena_rs_nr.jpg', im_rs_nr)
    cv.imshow('lena_rs_ln.jpg', im_rs_ln)
    cv.imshow('lena_rs_cb.jpg', im_rs_cb)
    cv.imshow('lena_rs_lz.jpg', im_rs_lz)

    # 沿y轴翻转图像（水平翻转）
    im_flip = cv.flip(im, 1)
    cv.imshow('lena_flip.jpg', im_flip)

    # 以图像中心为旋转点旋转图像
    h, w = im.shape[:2]
    M = cv.getRotationMatrix2D((w/2, h/2), 45, 1)
    im_rt = cv.warpAffine(im, M, (w, h))
    cv.imshow('lena_rt.jpg', im_rt)

    # 沿x轴负方向移动100个像素
    x = -100
    y = 0
    M = np.float32([[1, 0, x],[0, 1, y]])
    im_trans = cv.warpAffine(im, M, (w, h))
    cv.imshow('lena_trans.jpg', im_trans)

    cv.waitKey()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
