#!/usr/bin/env python3
# encoding:utf-8

import time
import argparse
import numpy as np
import cv2 as cv
from arm_sdk.scservo import *
from arm_sdk.three_inverse_kinematics import step

# 舵机编号，抓手为1，底盘为6
SCS_ID_1 = 1  # SCServo ID : 1  抓手开合
SCS_ID_2 = 2  # SCServo ID : 2  抓手旋转
SCS_ID_3 = 3  # SCServo ID : 3  第三连杆
SCS_ID_4 = 4  # SCServo ID : 4  第二连杆
SCS_ID_5 = 5  # SCServo ID : 5  第一连杆
SCS_ID_6 = 6  # SCServo ID : 6  控制整个机械臂旋转
SCS_MOVING_SPEED    = 1000  # SCServo moving speed 旋转速度
SCS_MOVING_ACC      = 50    # SCServo moving acc   旋转加速度

BEST_DETECT_FRAME_WIDTH    = 320
BEST_DETECT_FRAME_HEIGHT   = 240

def main(device_id=0, port_name='/dev/ttyUSB0'):
    flag, packetHandler, portHandler = arm_init(port_name)
    if flag == False:
        return

    # 打开深度相机
    # 如果失败，修改device_id为0，1，2等中的某个值，继续尝试
    # 仍然失败，检查USB摄像头是否连接
    orbbec_cap = cv.VideoCapture(device_id, cv.CAP_OBSENSOR)
    if orbbec_cap.isOpened() == False:
        print('打开深度相机失败。')
        return

    # 创建一个窗口，用于显示图像
    cv.namedWindow('Camera', cv.WINDOW_AUTOSIZE)

    # 装载人脸检测ONNX模型
    detector = cv.FaceDetectorYN.create(
        './models/face_detection_yunet_2023mar.onnx',
        '',
        (BEST_DETECT_FRAME_WIDTH, BEST_DETECT_FRAME_HEIGHT), # 输入图像尺寸
        score_threshold=0.9, # 阈值，应<1，越大误检测越少
        backend_id=cv.dnn.DNN_BACKEND_TIMVX, # 使用TIMVX后端，如果不适用NPU加速，而使用CPU计算，注释掉此行及下一行
        target_id=cv.dnn.DNN_TARGET_NPU # 使用NPU
    )

    # 循环，碰到按键盘就退出
    while cv.waitKey(1) < 0:
        #读帧图像
        if orbbec_cap.grab():
            # 解码grab()获取的帧数据
            # rgb数据
            ret_bgr, frame = orbbec_cap.retrieve(None, cv.CAP_OBSENSOR_BGR_IMAGE)
            if ret_bgr:    
                h, w, _ = frame.shape

                # 在320宽度的图像上检测人脸效果最佳
                frame320 = cv.resize(frame, (BEST_DETECT_FRAME_WIDTH, BEST_DETECT_FRAME_HEIGHT))
                retval, faces = detector.detect(frame320)
                if faces is None:
                    faces = []
                else:
                    # 将检测结果乘以系数，使之为原始大图像的人脸位置
                    faces *= w / BEST_DETECT_FRAME_WIDTH

                for face in faces:

                    # 把人脸框画到图像上
                    bbox = face[0:4].astype(np.int32)
                    cv.rectangle(frame, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 255, 0), 2)

                    adjust_arm(packetHandler, face, (w, h)) # 转动机械臂对准人脸

                # 把结果图像显示到窗口里
                cv.imshow('Camera', frame)

    portHandler.closePort() 
    orbbec_cap.release()
    cv.destroyAllWindows()

def arm_init(port_name='/dev/ttyUSB0'):
    # 初始化机械臂
    portHandler = PortHandler(port_name)

    if not portHandler.openPort():
        print('打开串口失败。')
        return False, None, None

    baudrate = 500000
    if not portHandler.setBaudRate(baudrate):
        print('设置波特率失败。')
        portHandler.closePort()
        return False, None, None

    packetHandler = sms_sts(portHandler)
    # 初始化机械臂姿态为“站立”状态
    print('初始化机械臂调整为“站立”姿态。')
    scs_comm_result, scs_error = packetHandler.WritePosEx(SCS_ID_1, 2047, SCS_MOVING_SPEED, SCS_MOVING_ACC)
    scs_comm_result, scs_error = packetHandler.WritePosEx(SCS_ID_2, 2047, SCS_MOVING_SPEED, SCS_MOVING_ACC)
    scs_comm_result, scs_error = packetHandler.WritePosEx(SCS_ID_3, 3070, SCS_MOVING_SPEED, SCS_MOVING_ACC)
    scs_comm_result, scs_error = packetHandler.WritePosEx(SCS_ID_4, 2047, SCS_MOVING_SPEED, SCS_MOVING_ACC)
    scs_comm_result, scs_error = packetHandler.WritePosEx(SCS_ID_5, 2047, SCS_MOVING_SPEED, SCS_MOVING_ACC)
    scs_comm_result, scs_error = packetHandler.WritePosEx(SCS_ID_6, 2047, SCS_MOVING_SPEED, SCS_MOVING_ACC)
    time.sleep(2)

    return True, packetHandler, portHandler

# def arm_unload():

# 此demo中只调整舵机3和舵机6
def adjust_arm(packetHandler, face, frame_size):
    w = frame_size[0]
    h = frame_size[1]

    # 计算人脸框的中心
    centerx = face[0] + face[2] / 2  
    centery = face[1] + face[3] / 2 
    # 为了让人脸处于中心，摄像头在x和y方向应该移动的角度（移动1个像素，对应调整1step）
    stepx = int((centerx - w /2))
    stepy = int((centery - h /2))

    # 当前舵机3和舵机6的角度数值
    scs_present_position_3, scs_comm_result, scs_error = packetHandler.ReadPos(SCS_ID_3)
    scs_present_position_6, scs_comm_result, scs_error = packetHandler.ReadPos(SCS_ID_6)
    # 转动舵机3和舵机6到指定位置
    # 舵机3对应y方向，舵机6对应x方向
    scs_comm_result, scs_error = packetHandler.WritePosEx(SCS_ID_3, scs_present_position_3 + stepy, SCS_MOVING_SPEED, SCS_MOVING_ACC)
    scs_comm_result, scs_error = packetHandler.WritePosEx(SCS_ID_6, scs_present_position_6 + stepx, SCS_MOVING_SPEED, SCS_MOVING_ACC)
    time.sleep(0.01)


if __name__ == '__main__':
    argParser = argparse.ArgumentParser()
    # 串口：Windows: "COM1"  Linux: "/dev/ttyUSB0"  Mac: "/dev/tty.usbserial-*"
    argParser.add_argument('-i', '--id', type=int, default=0, help='摄像头ID号')
    argParser.add_argument('-n', '--name', type=str, default='/dev/ttyUSB0',help='串口端号')
    args = argParser.parse_args()

    main(args.id, args.name)
