#!/usr/bin/env python3
# encoding:utf-8

import time

from hiwonderhand import HiwonderHand

# 通过串口连接机械手掌
# Linux系统应该是 /dev/ttyUSB0 或类似
# macOS系统应该是 /dev/cu.usbserial-1110 或类似
robotHand = HiwonderHand('/dev/ttyUSB0', 9600)

# 设置机械手掌运行速度
robotHand.setMotionTime(1000)

# 转动手腕
robotHand.setWrist(0) #最左侧
time.sleep(2) #暂停两秒，留出时间看运动效果
robotHand.setWrist(1) #左右侧
time.sleep(2) #暂停两秒，留出时间看运动效果

# 伸直大拇指
robotHand.setFinger1(1)
time.sleep(2) #暂停两秒，留出时间看运动效果

# 伸直食指
robotHand.setFinger2(1)
time.sleep(2) #暂停两秒，留出时间看运动效果

# 半弯曲中指
robotHand.setFinger3(0.5)
time.sleep(2) #暂停两秒，留出时间看运动效果

# 弯曲无名指和小指
robotHand.setFinger4(0)
robotHand.setFinger5(0)
time.sleep(2) #暂停两秒，留出时间看运动效果

# 握拳，即所有手指全部弯曲
robotHand.setFingers(0, 0, 0, 0, 0)
time.sleep(2) #暂停两秒，留出时间看运动效果


# 张开手
robotHand.setFingers(1, 1, 1, 1, 1)
time.sleep(2) #暂停两秒，留出时间看运动效果

# 竖中指
robotHand.setFingers(0, 0, 1, 0, 0)
time.sleep(2) #暂停两秒，留出时间看运动效果
