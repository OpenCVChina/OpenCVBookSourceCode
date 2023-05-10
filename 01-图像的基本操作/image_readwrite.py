#!/usr/bin/env python3
# encoding:utf-8


import cv2 as cv

def main():

    # 读入图像
    im = cv.imread("lena.jpg")

    # 读入图像,同时转换为灰度图
    im_gray = cv.imread("lena.jpg", cv.IMREAD_GRAYSCALE)

    # 读入图像,同时将图像大小缩小为原始大小的1/2
    im_small = cv.imread("lena.jpg", cv.IMREAD_REDUCED_COLOR_2)
    # 将上面缩小的图像写入文件
    cv.imwrite("lena_small.jpg", im_small)

    # 显示图像
    cv.imshow("Lena", im)
    cv.imshow("Lena_Gray", im_gray)
    cv.imshow("Lena_small", im_small)

    cv.waitKey()
    cv.destroyAllWindows()


if __name__  == '__main__':
    main()
