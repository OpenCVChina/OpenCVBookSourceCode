#!/usr/bin/env python3
# encoding:utf-8


import numpy as np
import cv2 as cv


def main():
    # 读入图像
    img1 = cv.imread('box.png', cv.IMREAD_GRAYSCALE)          
    img2 = cv.imread('box_in_scene.png', cv.IMREAD_GRAYSCALE) 

    # 提取SIFT特征
    sift = cv.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # 特征匹配
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1, des2, k = 2)

    # 绘制，显示
    good = []
    for m, n in matches:
        if m.distance < 0.6 * n.distance:
            good.append([m])
    img3 = cv.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags = cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    cv.imshow('Matches', img3)
    cv.imwrite('matches.jpg', img3)
    cv.waitKey()
    cv.destroyAllWindows()



if __name__ == '__main__':
    main()
