#!/usr/bin/env python3
# encoding:utf-8

import numpy as np 
import cv2 as cv


def main():
    # 打开深度相机
    orbbec_cap = cv.VideoCapture(0, cv.CAP_OBSENSOR)
    if orbbec_cap.isOpened() == False:
        print("Fail to open obsensor capture.")
        exit(0)

    while True:
        # 从相机获取帧数据
        if orbbec_cap.grab():
            # print("Grabbing data succeeds.")

            # 解码grab()获取的帧数据
            # rgb数据
            ret_bgr, bgr_image = orbbec_cap.retrieve(None, cv.CAP_OBSENSOR_BGR_IMAGE)
            if ret_bgr:
                cv.imshow("BGR", bgr_image)
            # 深度数据
            ret_depth, depth_map = orbbec_cap.retrieve(None, cv.CAP_OBSENSOR_DEPTH_MAP)
            if ret_depth:
                # 为了在屏幕上显示深度图，需要额外做一些处理
                color_depth_map = cv.normalize(depth_map, None, 0, 255, cv.NORM_MINMAX, cv.CV_8UC1)
                color_depth_map = cv.applyColorMap(color_depth_map, cv.COLORMAP_JET)
                cv.imshow("DEPTH",  color_depth_map)

            # print("ret_bgr: {} ret_depth: {}".format(ret_bgr, ret_depth))

        else:
            print("Fail to grab data from camera.")


        if cv.pollKey() >= 0:
            break

    orbbec_cap.release()


if __name__ == '__main__':
    main()