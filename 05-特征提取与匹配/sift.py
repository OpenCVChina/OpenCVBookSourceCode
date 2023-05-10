#!/usr/bin/env python3
# encoding:utf-8


import cv2 as cv
import numpy as np


def main():

    # 读入图像
    img = cv.imread('box_in_scene.png')
    cv.imshow('box_in_scene', img)
    # 将图像转为灰度图
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    sift = cv.SIFT_create()
    # 计算SIFT关键点
    kp = sift.detect(gray, None)

    # 绘制关键点
    # cv.drawKeypoints(gray, kp, img)
    # cv.imshow('sift_keypoints_1', img)
    # cv.imwrite('sift_keypoints_1.jpg', img)
    # 绘制关键点（大小和方向）
    cv.drawKeypoints(gray, kp, img, flags = cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv.imshow('sift_keypoints_2', img)
    cv.imwrite('sift_keypoints_2.jpg', img)

    cv.waitKey()
    cv.destroyAllWindows()



if __name__ == '__main__':
    main()
