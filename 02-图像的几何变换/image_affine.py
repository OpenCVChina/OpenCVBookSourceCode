#!/usr/bin/env python3
# encoding:utf-8


import cv2 as cv
import numpy as np


def main():

    # 读入图像
    im = cv.imread('lena.jpg')
    cv.imshow('lena.jpg', im)

    # 用原始图像和目标图像中三对对应点计算仿射变换矩阵 
    srcTri = np.float32([[0, 0], [im.shape[1] - 1, 0], [0, im.shape[0] - 1]])
    dstTri = np.float32([[0, im.shape[1] * 0.3], [im.shape[1] * 0.9, im.shape[0] * 0.2], [im.shape[1] * 0.1, im.shape[0] * 0.7]])
    warp_mat = cv.getAffineTransform(srcTri, dstTri)

    # 对原始图像进行仿射变换得到目标图像
    im_affine = cv.warpAffine(im, warp_mat, (im.shape[1], im.shape[0]))
    cv.imshow('im_affine.jpg', im_affine)

    cv.waitKey()
    cv.destroyAllWindows()



if __name__ == '__main__':
    main()
