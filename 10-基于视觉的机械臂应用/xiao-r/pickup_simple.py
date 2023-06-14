#!/usr/bin/env python3
# encoding:utf-8

import argparse
import numpy as np
import time 
import threading
import cv2 as cv
# from nanodet import NanoDet
from arm_sdk.scservo import *
from arm_sdk.three_inverse_kinematics import step


HS_TH_LOWER = np.array([0, 180, 0])         # 目标药瓶最低hs值
HS_TH_UPPER = np.array([30, 255, 255])      # 目标药瓶最高hs值
BOTTLE_WIDTH    = 50    # 药瓶的实际宽度为50mm
END_CAM_X       = 100   # 执行器末端与相机的水平距离为100mm
END_O_X         = 30    # 执行器末端与底座中心的水平距离为30mm

# 舵机编号，抓手为1，底盘为6
SCS_ID_1 = 1  # SCServo ID : 1  抓手开合
SCS_ID_2 = 2  # SCServo ID : 2  抓手旋转
SCS_ID_3 = 3  # SCServo ID : 3  第三连杆
SCS_ID_4 = 4  # SCServo ID : 4  第二连杆
SCS_ID_5 = 5  # SCServo ID : 5  第一连杆
SCS_ID_6 = 6  # SCServo ID : 6  控制整个机械臂旋转
SCS_MOVING_SPEED    = 500   # SCServo moving speed 旋转速度
SCS_MOVING_ACC      = 40    # SCServo moving acc   旋转加速度

# 橙色小药瓶放置于某一高度水平平台且在相机/夹手正前方，
# 机械臂根据药瓶在相机视野中的位置调整姿态将其抓取后调整自身呈“站立”姿态，然后再“坐下”
def main():
    # 初始化机械臂
    portHandler = PortHandler('/dev/ttyUSB0')

    if not portHandler.openPort():
        print('打开串口失败。')
        return

    baudrate = 500000
    if not portHandler.setBaudRate(baudrate):
        print('设置波特率失败。')
        portHandler.closePort()
        return

    packetHandler = sms_sts(portHandler)
    # 初始化机械臂姿态为“坐立”状态
    print('将机械臂调整为“坐立”姿态。')
    arm_initpose(packetHandler)
    time.sleep(2)

    grab = True

    # 打开深度相机
    orbbec_cap = cv.VideoCapture(0, cv.CAP_OBSENSOR)
    if orbbec_cap.isOpened() == False:
        print('打开深度相机失败。')
        exit(1)

    frame_idx = 0
    dist_list = []

    while True:
        # 从相机获取帧数据
        if orbbec_cap.grab():
            frame_idx += 1

            # 解码grab()获取的帧数据
            # rgb数据
            ret_bgr, bgr_image = orbbec_cap.retrieve(None, cv.CAP_OBSENSOR_BGR_IMAGE)
            # 深度数据
            ret_depth, depth_map = orbbec_cap.retrieve(None, cv.CAP_OBSENSOR_DEPTH_MAP)
            if ret_bgr:    
                h, w, _ = bgr_image.shape
      
                # 采集图像100帧后开始目标检测和抓取
                if frame_idx > 100 and grab and ret_depth:
                    # 检测要抓取的目标
                    bbox = detection(bgr_image)
                    if bbox is not None: 
                        # 在图像上绘制目标的矩形框
                        cv.rectangle(bgr_image, bbox[0], bbox[1], (0, 255, 0))  

                        # 计算目标矩形框中心在图像中的坐标
                        oxy = (bbox[1] + bbox[0]) / 2
                                            # 在深度图中获取oxy的值作为目标与相机的距离
                        if int(oxy[0]) >=0 and int(oxy[0]) <= (w -1) and int(oxy[1]) >=0 and int(oxy[1]) <= (h -1):
                            dist = depth_map[int(oxy[1]), int(oxy[0])]
                            print('dist: {}mm'.format(dist))                            
                            # Gemini2深度范围0.15m-10m
                            if dist > 150 and dist < 10000: 
                                dist_list.append(dist)
                                # 目前Gemini2在某些情况下输出的深度值不稳定
                                # 需要从多次深度数据来获取oxy的最终值
                                if len(dist_list) == 20:
                                    dist_list.sort(reverse=True)
                                    # print(dist_list[0:10])
                                    dist_avg = np.average(dist_list[0:10]) 
                                    print('avg dist: {}mm'.format(dist_avg)) 
                                    # 调整机械臂位置，使得目标位于图像中间

                                    # 创建一个新的线程来控制机械臂抓取目标
                                    arm_thread = threading.Thread(target=pickup, args=(dist_avg, 130, packetHandler))   # 瓶子中心距机械臂底座约130mm
                                    arm_thread.start()
                                    grab = False
                                    dist_list.clear()
                    else:
                        print('未检测到目标物。')
                cv.putText(bgr_image, 'frame {}'.format(frame_idx), (20, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
                cv.imshow('Camera', bgr_image)
        else:
            print('未获取到帧数据。')

        if cv.pollKey() >= 0:
            break

    cv.destroyAllWindows()
    orbbec_cap.release()
    portHandler.closePort()

def detection(image):
    bbox = None
    hsv_img = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    # 二值化
    mask = cv.inRange(hsv_img, HS_TH_LOWER, HS_TH_UPPER)
    mask = cv.dilate(mask, None)
    mask = cv.dilate(mask, None)
    cv.imshow('mask', mask)
    # 寻找连通区域
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    max_area = 0
    max_idx = -1
    # 取面积最大的连通区域为目标
    for idx, contour in enumerate(contours):
        area = cv.contourArea(contour)
        if area > max_area:
            max_area = area
            max_idx = idx    
    if max_idx > -1:
        # 目标的矩形框
        x, y, w, h = cv.boundingRect(contours[max_idx]) 
        bbox = (np.array([[x, y], [x + w, y + h]])).astype(np.int32)

    return bbox

def arm_initpose(phandler):
    # “坐立”姿态，夹手先张开再闭合
    scs_comm_result, scs_error = phandler.WritePosEx(SCS_ID_1, 1600, SCS_MOVING_SPEED, SCS_MOVING_ACC)
    scs_comm_result, scs_error = phandler.WritePosEx(SCS_ID_2, 2047, SCS_MOVING_SPEED, SCS_MOVING_ACC)
    scs_comm_result, scs_error = phandler.WritePosEx(SCS_ID_3, 3070, SCS_MOVING_SPEED, SCS_MOVING_ACC)
    scs_comm_result, scs_error = phandler.WritePosEx(SCS_ID_4, 1023, SCS_MOVING_SPEED, SCS_MOVING_ACC)
    scs_comm_result, scs_error = phandler.WritePosEx(SCS_ID_5, 3070, SCS_MOVING_SPEED, SCS_MOVING_ACC)
    scs_comm_result, scs_error = phandler.WritePosEx(SCS_ID_6, 2047, SCS_MOVING_SPEED, SCS_MOVING_ACC)
    time.sleep(2)
    scs_comm_result, scs_error = phandler.WritePosEx(SCS_ID_1, 2300, SCS_MOVING_SPEED, SCS_MOVING_ACC)

def arm_sitdown(phandler):
    # 保持夹手和底座姿态，机械臂“坐下来”
    scs_comm_result, scs_error = phandler.WritePosEx(SCS_ID_3, 3070, SCS_MOVING_SPEED, SCS_MOVING_ACC)
    scs_comm_result, scs_error = phandler.WritePosEx(SCS_ID_4, 1023, SCS_MOVING_SPEED, SCS_MOVING_ACC)
    scs_comm_result, scs_error = phandler.WritePosEx(SCS_ID_5, 3070, SCS_MOVING_SPEED, SCS_MOVING_ACC)

def arm_standup(phandler):
    # 保持夹手和底座姿态，机械臂“站起来”
    scs_comm_result, scs_error = phandler.WritePosEx(SCS_ID_3, 3070, SCS_MOVING_SPEED, SCS_MOVING_ACC)
    scs_comm_result, scs_error = phandler.WritePosEx(SCS_ID_4, 2047, SCS_MOVING_SPEED, SCS_MOVING_ACC)
    scs_comm_result, scs_error = phandler.WritePosEx(SCS_ID_5, 2047, SCS_MOVING_SPEED, SCS_MOVING_ACC)

def arm_set_links(step_3, step_4, step_5, phandler):
    scs_comm_result, scs_error = phandler.WritePosEx(SCS_ID_3, step_3, SCS_MOVING_SPEED, SCS_MOVING_ACC)
    scs_comm_result, scs_error = phandler.WritePosEx(SCS_ID_4, step_4, SCS_MOVING_SPEED, SCS_MOVING_ACC)
    scs_comm_result, scs_error = phandler.WritePosEx(SCS_ID_5, step_5, SCS_MOVING_SPEED, SCS_MOVING_ACC)

def arm_set_end(step_1, phandler):
    scs_comm_result, scs_error = phandler.WritePosEx(SCS_ID_1, step_1, SCS_MOVING_SPEED, SCS_MOVING_ACC)

def pickup(xo, yo, phandler):
    print('===== 机械臂开始抓取目标药瓶 =====')
    # 末端执行器的坐标
    xe_1 = xo - END_CAM_X + END_O_X
    xe_2 = xe_1 + (BOTTLE_WIDTH + 10)
    ye = yo

    # 计算末端执行器运动至目标物前舵机3、4 5的角度数值
    step_3, step_4, step_5 = step(xe_1, yo)
    if step_3 < 1000 or step_3 > 3200 or step_4 < 540 or step_4 > 3400 or step_5 < 1000 or step_5 > 3050:
        print('舵机无法转到所需角度。')
        return False
    # 先将末端执行器运动至目标物前 
    arm_set_links(step_3, step_4, step_5, phandler)
    time.sleep(3)
    # 然后张开夹手  
    arm_set_end(1600, phandler)
    time.sleep(1)

    # 再计算末端执行器夹住目标物时舵机3、4、5的角度数值
    step_3, step_4, step_5 = step(xe_2, yo)
    if step_3 < 1000 or step_3 > 3200 or step_4 < 540 or step_4 > 3400 or step_5 < 1000 or step_5 > 3050:
        print('舵机无法转到所需角度。')
        return False
    # 将末端执行器运动至夹住目标物时的位置 
    arm_set_links(step_3, step_4, step_5, phandler)
    time.sleep(3)
    # 然后闭合夹手  
    arm_set_end(2300, phandler)
    time.sleep(2)

    # 夹住目标物后机械臂调整呈“站立”姿态
    arm_standup(phandler)
    time.sleep(5)
    # 机械臂“坐下去”
    arm_sitdown(phandler)
    time.sleep(5)
    print('===== 机械臂完成抓取目标药瓶 =====')


if __name__ == '__main__':
    main()
