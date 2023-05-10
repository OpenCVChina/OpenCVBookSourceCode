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

    # FLANN参数
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)   
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k = 2)

    # 只绘制好的匹配，所以创建一个掩膜
    matchesMask = [[0,0] for i in range(len(matches))]
    # 比率测试
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            matchesMask[i] = [1, 0]
    draw_params = dict(matchColor = (0, 255, 0),
                       singlePointColor = (0, 0, 255),
                       matchesMask = matchesMask,
                       flags = cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)    
    img3 = cv.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)

    cv.imshow('Matches', img3)
    cv.imwrite('flann_match.jpg', img3)
    cv.waitKey()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
