#!/usr/bin/env python3
# encoding:utf-8


import cv2 as cv
import numpy as np


def main():

    # 读入图像
    im = cv.imread('blox.jpg')
    # 将图像转为灰度图
    im_grey = cv.cvtColor(im, cv.COLOR_BGR2GRAY)

    # Sobel边缘检测。注意输出图像的数据类型
    im_sobel_x = cv.Sobel(im_grey, cv.CV_16S, 1, 0, 3)
    im_sobel_y = cv.Sobel(im_grey, cv.CV_16S, 0, 1, 3)
    im_sobel_x = cv.convertScaleAbs(im_sobel_x)
    im_sobel_y = cv.convertScaleAbs(im_sobel_y) 
    # 合并x方向和y方向的边缘  
    im_sobel = cv.addWeighted(im_sobel_x, 0.5, im_sobel_y, 0.5, 0) 

    # Scharr边缘检测。注意输出图像的数据类型
    im_scharr_x = cv.Scharr(im_grey, cv.CV_16S, 1, 0, 3)
    im_scharr_y = cv.Scharr(im_grey, cv.CV_16S, 0, 1, 3)
    im_scharr_x = cv.convertScaleAbs(im_scharr_x)
    im_scharr_y = cv.convertScaleAbs(im_scharr_y)   
    # 合并x方向和y方向的边缘
    im_scharr = cv.addWeighted(im_scharr_x, 0.5, im_scharr_y, 0.5, 0)    

    # Laplacian边缘检测。注意输出图像的数据类型
    im_laplacian_1  = cv.Laplacian(im_grey, cv.CV_32F)
    # 设置卷积核大小
    im_laplacian_3 = cv.Laplacian(im_grey, cv.CV_32F, 7)
    im_laplacian_1 = cv.convertScaleAbs(im_laplacian_1)   
    im_laplacian_3 = cv.convertScaleAbs(im_laplacian_3)

    cv.imshow('blox.jpg', im)
    cv.imshow('blox_sobel_x.jpg', im_sobel_x)
    cv.imshow('blox_sobel_y.jpg', im_sobel_y)
    cv.imshow('blox_sobel.jpg', im_sobel)
    cv.imshow('blox_scharr.jpg', im_scharr)
    cv.imshow('blox_laplacian1.jpg', im_laplacian_1)
    cv.imshow('blox_laplacian3.jpg', im_laplacian_3)
    cv.waitKey()
    cv.destroyAllWindows()



if __name__ == '__main__':
    main()
