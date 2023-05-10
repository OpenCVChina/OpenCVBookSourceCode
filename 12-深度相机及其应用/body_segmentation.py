#!/usr/bin/env python3
# encoding:utf-8

import numpy as np 
import cv2 as cv

window_name = "Distance"
thres = [700, 900]  # 距离阈值(mm)

def set_thre1(thre):
    thres[0] = thre

def set_thre2(thre):
    thres[1] = thre

def main():
    cv.namedWindow(window_name)
    # 创建滚动条以便手动调节距离阈值
    cv.createTrackbar("dist1", window_name, 700, 1000, set_thre1)
    cv.createTrackbar("dist2", window_name, 900, 1000, set_thre2)

    # 打开深度相机
    orbbec_cap = cv.VideoCapture(0, cv.CAP_OBSENSOR)
    if orbbec_cap.isOpened() == False:
        print("Fail to open obsensor capture!")
        exit(0)

    while cv.waitKey(1) < 0:
        # 从相机获取帧数据
        if orbbec_cap.grab():
            # 解码grab()获取的帧数据
            # rgb数据
            ret_bgr, bgr_image = orbbec_cap.retrieve(flag=cv.CAP_OBSENSOR_BGR_IMAGE)

            # 深度数据（单位mm）
            ret_depth, depth_map = orbbec_cap.retrieve(flag=cv.CAP_OBSENSOR_DEPTH_MAP)
            if ret_bgr and ret_depth:
                # 为了在屏幕上显示深度图，需要额外做一些处理
                color_depth_map = cv.normalize(depth_map, None, 0, 255, cv.NORM_MINMAX, cv.CV_8UC1)
                color_depth_map = cv.applyColorMap(color_depth_map, cv.COLORMAP_JET)
                cv.imshow("Depth",  color_depth_map)
                cv.imwrite("depth.jpg", color_depth_map)
                
                # 根据设置的距离阈值thres把深度图二值化
                segment_image, body_contour = segment_body(depth_map, thres)
                cv.imshow("Segmentation", segment_image)
                cv.imwrite("segment.jpg", segment_image)
                if body_contour is not None:
                    # 将分割的轮廓绘制在RGB图上
                    cv.drawContours(bgr_image, body_contour, -1, (0, 255, 0), 2, cv.LINE_AA)
                cv.imshow("Body Contour", bgr_image)
                cv.imwrite("contour.jpg", bgr_image)
        else:
            print("Fail to grab data from camera!")
    
    orbbec_cap.release()
        

def segment_body(depth_image, thres):
    # 人体与相机的距离在thres[0]和thres[1]之间
    output = cv.inRange(depth_image, thres[0], thres[1])
    output = cv.dilate(output, None)
    output = cv.dilate(output, None)

    # 寻找连通区域
    contours, _ = cv.findContours(output, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    max_area = 0
    max_idx = -1
    # 取面积最大的连通区域为人体
    for idx, contour in enumerate(contours):
        area = cv.contourArea(contour)
        if area > max_area:
            max_area = area
            max_idx = idx    
    body_contour = None
    if max_idx > -1:
        body_contour = (contours[max_idx],) # 分割出的人体的轮廓
        output = np.zeros(depth_image.shape, dtype=np.uint8) # 分割出的人体mask图像
        cv.fillPoly(output, body_contour, 255)

    return output, body_contour


if __name__ == '__main__':
    main()
