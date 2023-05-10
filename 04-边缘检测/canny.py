#!/usr/bin/env python3
# encoding:utf-8


import cv2 as cv

window_name = 'edge map'

def canny(low_threshold):

    # 此处固定threshold2等于3xthreshold1
    high_threshold = low_threshold * 3
    kernel_size = 3

    im = cv.imread('building.jpg')
    # 将图像转为灰度图
    im_grey = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    # 对图像进行高斯滤波平滑图像
    im_blur = cv.GaussianBlur(im_grey, (3, 3), 0, 0)

    # Canny边缘检测
    edges = cv.Canny(im_blur, low_threshold, high_threshold, kernel_size)
    # 以原始图像的色调显示边缘
    mask = edges != 0
    edge_map = im * (mask[:,:,None].astype(im.dtype))

    cv.imshow(window_name, edge_map)


def main():
    
    max_lowThreshold = 100
    cv.namedWindow(window_name)
    cv.createTrackbar('low threshold', window_name, 0, max_lowThreshold, canny)
    canny(0)
    cv.waitKey()
    cv.destroyAllWindows()



if __name__ == '__main__':
    main()
