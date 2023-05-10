#!/usr/bin/env python3
# encoding:utf-8


import time
import cv2 as cv
from RobotArm import RobotArm

window_name = 'Camera'
servo_id = 1
arm = RobotArm("/dev/ttyUSB0", 115200)

def set_id(idx):
    global servo_id 
    servo_id = idx

def set_angle(angle):
    arm.setAngle(servo_id, angle-120, 100)
    time.sleep(0.1)

def main():

    # 初始化机械臂姿势，让机械臂呈站立姿态
    print('初始化机械臂...')
    arm.setAngle(1, 0, 1000) # 电机1，控制机械爪，1000ms完成调整
    time.sleep(1.0) # 暂停1.0秒，等待电机调整完毕
    print('舵机1 ok')
    arm.setAngle(2, 0, 1000)
    time.sleep(1.0) # 暂停1.0秒，等待电机调整完毕
    print('舵机2 ok')
    arm.setAngle(3, 0, 1000) # 电机3，控制摄像头
    time.sleep(1.0) # 暂停1.0秒，等待电机调整完毕
    print('舵机3 ok')
    arm.setAngle(4, 0, 1000)
    time.sleep(1.0) # 暂停1.0秒，等待电机调整完毕
    print('舵机4 ok')
    arm.setAngle(5, 0, 1000) 
    time.sleep(1.0) # 暂停1.0秒，等待电机调整完毕
    print('舵机5 ok')
    arm.setAngle(6, 0, 1000) 
    time.sleep(1.0) # 暂停1.0秒，等待电机调整完毕
    print('舵机6 ok')

    cv.namedWindow(window_name)
    cv.createTrackbar('servo id', window_name, 1, 6, set_id)
    cv.createTrackbar('angle', window_name, 0, 240, set_angle)

    cap = cv.VideoCapture(0)
    while cv.waitKey(1) < 0:
        _, frame = cap.read()

        cv.imshow(window_name, frame)
    
    cv.destroyAllWindows()
    #退出程序前，将所有马达卸力
    arm.unloadBusServo(1) #马达1卸载动力
    arm.unloadBusServo(2) #马达2卸载动力
    arm.unloadBusServo(3) #马达3卸载动力
    arm.unloadBusServo(4) #马达4卸载动力
    arm.unloadBusServo(5) #马达5卸载动力
    arm.unloadBusServo(6) #马达6卸载动力



if __name__ == '__main__':
    main()
