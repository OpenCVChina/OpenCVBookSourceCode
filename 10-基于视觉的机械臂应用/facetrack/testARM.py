#!/usr/bin/env python3
# encoding:utf-8

import time
from RobotArm import RobotArm

arm = RobotArm("/dev/ttyUSB0", 115200)

# 初始化机械臂姿势，让机械臂呈站立姿态；顶部转90度，让摄像头平视
movetime = 2000 # 毫秒
movetime_short = 200 # 毫秒
arm.setAngle(1, -90, movetime) # 电机1的角度为-90（张开爪子），200ms完成调整
time.sleep(movetime/1000) # 暂停X秒，等待电机调整完毕
arm.setAngle(2, 0, movetime)
time.sleep(movetime/1000)
arm.setAngle(3, -40, movetime) # 电机3转到-90度，让摄像头平视
time.sleep(movetime/1000)
arm.setAngle(4, 0, movetime)
time.sleep(movetime/1000)
arm.setAngle(5, 0, movetime) 
time.sleep(movetime/1000)
arm.setAngle(6, 0, movetime)
time.sleep(movetime/1000)  


for motor in range(1, 7):
    arm.setAngle(motor, -20, movetime)
    for angle in range(-20, 20, 2):
        arm.setAngle(motor, angle, movetime_short)
        time.sleep(movetime_short/1000)
    arm.setAngle(motor, 0, movetime)
    time.sleep(movetime/1000)

arm.setAngle(1, -90, movetime) #张开爪子
time.sleep(movetime/1000)  

#退出程序前，将所有马达卸力
arm.unloadBusServo(1)
arm.unloadBusServo(2)
arm.unloadBusServo(3)
arm.unloadBusServo(4)
arm.unloadBusServo(5)
arm.unloadBusServo(6)        
