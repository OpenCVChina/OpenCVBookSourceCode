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

    orb = cv.ORB.create()
    # 计算ORB关键点
    kp = orb.detect(gray, None)

    # 绘制关键点
    cv.drawKeypoints(gray, kp, img)
    cv.imshow('orb_keypoints_1', img)
    cv.imwrite('orb_keypoints_1.jpg', img)
    # 绘制关键点（大小和方向）
    # cv.drawKeypoints(gray, kp, img, flags = cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # cv.imshow('orb_keypoints_2', img)
    # cv.imwrite('orb_keypoints_2.jpg', img)

    cv.waitKey()
    cv.destroyAllWindows()



if __name__ == '__main__':
    main()
