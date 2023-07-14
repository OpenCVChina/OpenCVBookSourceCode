#!/usr/bin/env python3
# encoding:utf-8

import argparse
import numpy as np
import time 
import threading
import cv2 as cv
from arm_sdk.scservo import *
from arm_sdk.three_inverse_kinematics import step


HS_TH_LOWER = np.array([0, 180, 0])         # 目标药瓶最低hs值
HS_TH_UPPER = np.array([30, 255, 255])      # 目标药瓶最高hs值
BOTTLE_WIDTH    = 50    # 药瓶的实际宽度为50mm
L0_CAM_X        = 50    # “坐立”姿态时底座中心与相机的水平距离为50mm
END_L0_Y         = 125   # “坐立”姿态时执行器末端与底座的竖直距离为125mm

# 舵机编号，抓手为1，底盘为6
SCS_ID_1 = 1  # SCServo ID : 1  抓手开合
SCS_ID_2 = 2  # SCServo ID : 2  抓手旋转
SCS_ID_3 = 3  # SCServo ID : 3  第三连杆
SCS_ID_4 = 4  # SCServo ID : 4  第二连杆
SCS_ID_5 = 5  # SCServo ID : 5  第一连杆
SCS_ID_6 = 6  # SCServo ID : 6  控制整个机械臂旋转
SCS_MOVING_SPEED    = 800   # SCServo moving speed 旋转速度
SCS_MOVING_ACC      = 40    # SCServo moving acc   旋转加速度

# 橙色小药瓶放置于某一高度水平平台且在相机/夹手正前方，
# 机械臂根据药瓶在相机视野中的位置调整姿态将其抓取后调整自身呈“站立”姿态，然后再“坐下”
def main():
    # 初始化机械臂
    portHandler = PortHandler('/dev/ttyUSB0')

    if not portHandler.openPort():
        print('打开串口失败！')
        return

    baudrate = 500000
    if not portHandler.setBaudRate(baudrate):
        print('设置波特率失败！')
        portHandler.closePort()
        return

    packetHandler = sms_sts(portHandler)
    # 初始化机械臂姿态为“坐立”状态
    print('* 将机械臂调整为“坐立”姿态。*')
    arm_initpose(packetHandler)
    time.sleep(2)

    grab = True

    # 打开深度相机
    orbbec_cap = cv.VideoCapture(0, cv.CAP_OBSENSOR)
    if orbbec_cap.isOpened() == False:
        print('打开深度相机失败！')
        exit(1)

    frame_idx = 0
    dist_list = []
    arm_thread = None

    print('* 开始采集图像，并从第100帧后开始检测目标并抓取。*')
    while True:
        if arm_thread is not None:
            if arm_thread.is_alive() == False:
                print('* 抓取完成，结束程序。*')
                break

        # 从相机获取帧数据
        if orbbec_cap.grab():
            frame_idx += 1

            # 解码grab()获取的帧数据
            # rgb数据
            ret_bgr, bgr_image = orbbec_cap.retrieve(None, cv.CAP_OBSENSOR_BGR_IMAGE)
            # 深度数据
            ret_depth, depth_map = orbbec_cap.retrieve(None, cv.CAP_OBSENSOR_DEPTH_MAP)
            if ret_depth:
                # 为了在屏幕上显示深度图，需要额外做一些处理
                color_depth_map = cv.normalize(depth_map, None, 0, 255, cv.NORM_MINMAX, cv.CV_8UC1)
                color_depth_map = cv.applyColorMap(color_depth_map, cv.COLORMAP_JET)
                cv.imshow("DEPTH",  color_depth_map)

            if ret_bgr:    
                h, w, _ = bgr_image.shape
      
                # 采集图像100帧后开始目标检测和抓取
                if frame_idx > 100 and grab and ret_depth:
                    # 检测要抓取的目标
                    bbox = detection(bgr_image)
                    if bbox is not None: 
                        # 计算目标矩形框中心在图像中的坐标
                        oxy = (bbox[1] + bbox[0]) / 2
                        # 计算目标与相机的距离，取目标中心小区域内深度图的平均值作为距离
                        x1 = int(oxy[0]) - 10
                        x2 = int(oxy[0]) + 10
                        y1 = int(oxy[1]) - 10
                        y2 = int(oxy[1]) + 10
                        roi = depth_map[y1:y2, x1:x2]
                        dist_avg = cv.sumElems(roi)[0] / (cv.countNonZero(roi) + 0.000001)
                        print('物体平均距离: {:.2f}mm'.format(dist_avg)) 
                        # Gemini2深度范围0.15m-10m，但实际上0.35m以内就无法准确测距了
                        # 在此距离外，相机可以较稳定准确的测距，但是机械臂长的范围又有限
                        # 目标距相机太远，机械臂无法触及
                        if dist_avg > 150 and dist_avg < 330:
                            # 在图像上绘制目标的绿色矩形框，表示物体摆放距离合适
                            cv.rectangle(bgr_image, bbox[0], bbox[1], (0, 255, 0), 2)  
                            # 创建一个新的线程来控制机械臂抓取目标
                            arm_thread = threading.Thread(target=pickup, args=(dist_avg, END_L0_Y, packetHandler))
                            arm_thread.start()
                            grab = False
                        elif dist_avg < 150:
                            print('请将抓取目标摆放远一点！')
                            # 在图像上绘制目标的红色矩形框，表示物体摆放距离不合适
                            cv.rectangle(bgr_image, bbox[0], bbox[1], (0, 0, 255), 2)  
                        else:
                            print('请将抓取目标摆放近一点！')
                            # 在图像上绘制目标的红色矩形框，表示物体摆放距离不合适
                            cv.rectangle(bgr_image, bbox[0], bbox[1], (0, 0, 255), 2)  
                    else:
                        print('未检测到目标物体！')
                cv.putText(bgr_image, 'frame {}'.format(frame_idx), (20, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
                cv.imshow('RGB', bgr_image)
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
    # cv.imshow('MASK', mask)
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
    # 添加延时实现分步执行，解决收回抖动问题
    scs_comm_result, scs_error = phandler.WritePosEx(SCS_ID_5, 2559, SCS_MOVING_SPEED, SCS_MOVING_ACC)  
    time.sleep(1)   
    scs_comm_result, scs_error = phandler.WritePosEx(SCS_ID_4, 1535, SCS_MOVING_SPEED, SCS_MOVING_ACC)
    time.sleep(1)   
    scs_comm_result, scs_error = phandler.WritePosEx(SCS_ID_3, 3070, SCS_MOVING_SPEED, SCS_MOVING_ACC)
    time.sleep(0.5) 
    scs_comm_result, scs_error = phandler.WritePosEx(SCS_ID_4, 1023, SCS_MOVING_SPEED, SCS_MOVING_ACC)
    time.sleep(1)   
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

# 控制夹手
def arm_set_end(step_1, phandler):
    if step_1 < 1400:
        step_1 = 1400
    if step_1 > 2700:
        step_1 = 2700
    # 闭合
    scs_comm_result, scs_error = phandler.WritePosEx(SCS_ID_1, step_1, SCS_MOVING_SPEED, SCS_MOVING_ACC)
    time.sleep(2)   # 预留足够的时间给机械臂执行到指定位置在读取电流，再根据电流判断力量
    # print("夹手舵机位置:",step_1)

    """ 检测方案一  直接减去某一个指定值，一次性减去 """
    # # 添加抓取时力度过大检测，检测电流如果电流抓取时电流过大，稍微把爪子张开一点
    # # 获取当前的电流和位置
    # scs_present_I, scs_present_position = phandler.ReadPosStatus(SCS_ID_1)
    # # 判断电流值，电流值越大说明爪子夹取的力量越大，越容易造成舵机烧坏
    # if scs_present_I > 60:
    #     scs_present_position = scs_present_position - 10   # 将当前的数值减去一定的值使爪子松开，减去的数值需要根据实际情况决定
    #     print("scs_present_position:",scs_present_position)
    #     phandler.WritePosEx(SCS_ID_1, scs_present_position, SCS_MOVING_SPEED, SCS_MOVING_ACC)

    """ 检测方案二  循环检测电流，分步减去较小的值，直到电流小于指定值 （建议使用该方案）"""
    while True:
        # 获取当前的电流和位置
        scs_present_I, scs_present_position = phandler.ReadPosStatus(SCS_ID_1)
        # print("抓取中：[ID:%03d] I:%2d PresPos:%d" % (SCS_ID_1, scs_present_I, scs_present_position))
        time.sleep(0.01)

        if scs_present_I >= 30:
            # print("抓取力量太大，进行松开调整")
            scs_present_position = scs_present_position - 1   # 将当前的数值减去一定的值使爪子松开
            # print("scs_present_position:",scs_present_position)
            phandler.WritePosEx(SCS_ID_1, scs_present_position, SCS_MOVING_SPEED, SCS_MOVING_ACC)
        else: 
            # print("力量符合！！")
            break

def pickup(xo, yo, phandler):
    print('===== 机械臂开始抓取目标药瓶 =====')
    print('请等待抓取过程结束...')
    # 末端执行器的坐标
    xe = xo - L0_CAM_X + 35 # 30mm为夹手为了夹住目标瓶子，需要从目标面前向前移动的水平距离
    ye = yo

    # 计算末端执行器运动至夹住目标物的位置舵机3、4 5的角度数值
    step_3, step_4, step_5 = step(xe, yo)
    if step_3 < 1000 or step_3 > 3200 or step_4 < 540 or step_4 > 3400 or step_5 < 1000 or step_5 > 3050:
        print('舵机无法转到所需角度。')
        return False

    # 张开夹手 
    arm_set_end(1600, phandler)
    # 将末端执行器运动至可以夹住目标物位置
    arm_set_links(step_3, step_4, step_5, phandler)
    time.sleep(4)
    
    # 然后闭合夹手  
    arm_set_end(2400, phandler)
    time.sleep(2)

    # 夹住目标物后机械臂调整呈“站立”姿态
    arm_standup(phandler)
    time.sleep(3)
    # 机械臂“坐下去”
    arm_sitdown(phandler)
    time.sleep(3)
    print('===== 机械臂完成抓取目标药瓶 =====')


if __name__ == '__main__':
    main()
