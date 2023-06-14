#!/usr/bin/env python3
# encoding:utf-8

import argparse
from arm_sdk.scservo import *
from arm_sdk.three_inverse_kinematics import step


# 舵机编号，抓手为1，底盘为6
SCS_ID_1 = 1  # SCServo ID : 1  抓手开合
SCS_ID_2 = 2  # SCServo ID : 2  抓手旋转
SCS_ID_3 = 3  # SCServo ID : 3  第三连杆
SCS_ID_4 = 4  # SCServo ID : 4  第二连杆
SCS_ID_5 = 5  # SCServo ID : 5  第一连杆
SCS_ID_6 = 6  # SCServo ID : 6  控制整个机械臂旋转

SCS_MOVING_SPEED    = 800   # SCServo moving speed 旋转速度
SCS_MOVING_ACC      = 50    # SCServo moving acc   旋转加速度


def main(serial_port_name='/dev/ttyUSB0', pose=2, index=None, step=None):

    # 初始化
    portHandler = PortHandler(serial_port_name)
    if not portHandler.openPort():
        print('打开串口失败。')
        return

    baudrate = 500000
    if not portHandler.setBaudRate(baudrate):
        print('设置波特率失败。')
        portHandler.closePort()
        return
    
    packetHandler = sms_sts(portHandler)
    if pose == 1: 
        print('调整机械臂呈“站立”姿态')
        scs_comm_result, scs_error = packetHandler.WritePosEx(SCS_ID_1, 2047, SCS_MOVING_SPEED, SCS_MOVING_ACC)
        scs_comm_result, scs_error = packetHandler.WritePosEx(SCS_ID_2, 2047, SCS_MOVING_SPEED, SCS_MOVING_ACC)
        scs_comm_result, scs_error = packetHandler.WritePosEx(SCS_ID_3, 3070, SCS_MOVING_SPEED, SCS_MOVING_ACC)
        scs_comm_result, scs_error = packetHandler.WritePosEx(SCS_ID_4, 2047, SCS_MOVING_SPEED, SCS_MOVING_ACC)
        scs_comm_result, scs_error = packetHandler.WritePosEx(SCS_ID_5, 2047, SCS_MOVING_SPEED, SCS_MOVING_ACC)
        scs_comm_result, scs_error = packetHandler.WritePosEx(SCS_ID_6, 2047, SCS_MOVING_SPEED, SCS_MOVING_ACC)

    elif pose == 2:
        print('调整机械臂呈“坐立”姿态')
        # 夹手舵机角度数值2300，保持闭合
        scs_comm_result, scs_error = packetHandler.WritePosEx(SCS_ID_1, 2300, SCS_MOVING_SPEED, SCS_MOVING_ACC)
        scs_comm_result, scs_error = packetHandler.WritePosEx(SCS_ID_2, 2047, SCS_MOVING_SPEED, SCS_MOVING_ACC)
        scs_comm_result, scs_error = packetHandler.WritePosEx(SCS_ID_3, 3070, SCS_MOVING_SPEED, SCS_MOVING_ACC)
        scs_comm_result, scs_error = packetHandler.WritePosEx(SCS_ID_4, 1023, SCS_MOVING_SPEED, SCS_MOVING_ACC)
        scs_comm_result, scs_error = packetHandler.WritePosEx(SCS_ID_5, 3070, SCS_MOVING_SPEED, SCS_MOVING_ACC)
        scs_comm_result, scs_error = packetHandler.WritePosEx(SCS_ID_6, 2047, SCS_MOVING_SPEED, SCS_MOVING_ACC)

    elif pose == 3:
        print('调整机械臂单个舵机')
        if index is None and step is None:
            print('请输入需要调整的舵机序号和其角度值。')
            portHandler.closePort()
            return
        # 舵机旋转角度数值，以2047为中间值
        scs_comm_result, scs_error = packetHandler.WritePosEx(index, step, SCS_MOVING_SPEED, SCS_MOVING_ACC)

    else:
        print('未定义模式。')

    portHandler.closePort()

if __name__ == '__main__':
    argParser = argparse.ArgumentParser()
    # 串口：Windows: "COM1"  Linux: "/dev/ttyUSB0"  Mac: "/dev/tty.usbserial-*"
    argParser.add_argument('-n', '--name', type=str, default='/dev/ttyUSB0', help='串口端号')
    argParser.add_argument('-p', '--pose', type=int, default=2, help='机械臂姿态。1-站立；2-坐立')
    argParser.add_argument('-i', '--index', type=int, default=None, help='舵机编号')
    argParser.add_argument('-s', '--step', type=int, default=None, help='舵机角度数值')
    args = argParser.parse_args()

    main(args.name, args.pose, args.index, args.step)
