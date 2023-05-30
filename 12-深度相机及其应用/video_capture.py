#!/usr/bin/env python3
# encoding:utf-8
# 奥比中光深度相机的RGB图像和深度图像获取和显示

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
                depth_map_8U = depth_map * 255.0 / 5000 # 将以毫米度量的距离归一化到0-255之间
                depth_map_8U = np.clip(depth_map_8U, 0, 255) # 超过255的值置为255
                depth_map_8U = np.uint8(depth_map_8U) #转成uint8格式
                cv.imshow("Depth: Gray", depth_map_8U) #以灰度图方式显示深度

                color_depth_map = cv.applyColorMap(depth_map_8U, cv.COLORMAP_JET) #转成伪彩色
                cv.imshow("Depth: ColorMap",  color_depth_map) #以伪彩色显示深度，视觉效果更好

            # print("ret_bgr: {} ret_depth: {}".format(ret_bgr, ret_depth))

        else:
            print("Fail to grab data from camera.")


        if cv.pollKey() >= 0:
            break

    orbbec_cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
