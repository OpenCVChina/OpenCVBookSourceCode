#!/usr/bin/env python3
# encoding:utf-8

import cv2 as cv
import numpy as np

def main():
    # 读入图像
    im_src = cv.imread('src.jpeg')
    im_dst = cv.imread('dst.jpeg')

    # 对应点对坐标
    src_points = np.array([[172, 884], [664, 522], [986, 1060], [802, 1442]])
    dst_points = np.array([[454, 388], [832, 468], [728, 788], [446, 842]])

    # 估计单应性变换矩阵
    H, _ = cv.findHomography(src_points, dst_points)

    # 对im_dst应用单应性变换矩阵，对比im_src观察变换后的图像
    h, w, _ = im_dst.shape
    im_dst_h = cv.warpPerspective(im_src, H, (w, h))

    # 显示图像
    cv.imshow('src image', im_src)
    cv.imshow('dst image', im_dst)
    cv.imshow('dst image transformed', im_dst_h)
    # 保存图像
    # cv.imwrite('dst_h.jpg', im_dst_h)
    cv.waitKey()

    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
