#!/usr/bin/env python3
# encoding:utf-8


import cv2 as cv
import numpy as np


def main():

    im_poisson = cv.imread('lena_poisson.jpg')

    # 3x3均值滤波
    im_average = cv.blur(im_poisson, (3, 3))

    # 高斯滤波
    im_gaussian3x3 = cv.GaussianBlur(im_poisson, (3, 3), 0, 0)
    im_gaussian5x5 = cv.GaussianBlur(im_poisson, (5, 5), 0, 0)
    im_gaussian3x3_2 = cv.GaussianBlur(im_poisson, (3, 3), 2, 2)

    im_sp = cv.imread('lena_sp.jpg')

    # 中值滤波
    im_median3x3 = cv.medianBlur(im_sp, 3)

    # 双边滤波
    im_bilateral = cv.bilateralFilter(im_poisson, 5, 80, 80)

    cv.imshow('lena_poisson.jpg', im_poisson)
    cv.imshow('lena_average.jpg', im_average)
    cv.imshow('lena_gaussian3x3.jpg', im_gaussian3x3)
    cv.imshow('lena_gaussian5x5.jpg', im_gaussian5x5)
    cv.imshow('lena_gaussian3x3_2.jpg', im_gaussian3x3_2)
    cv.imshow('lena_median3x3.jpg', im_median3x3)
    cv.imshow('lena_bilateral.jpg', im_bilateral)
    cv.waitKey()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
