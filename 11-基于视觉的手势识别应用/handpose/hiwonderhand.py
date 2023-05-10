#!/usr/bin/env python3
# encoding:utf-8

# 通过串口控制机械手掌
# 版本：2023年3月27日

import serial #Please install pyserial

class HiwonderHand:
    __serialHandle = 0
    __motionTimeByte0 = 0x00
    __motionTimeByte1 = 0x04

    def __init__(self, port="/dev/ttyUSB0", rate=9600):
        '''如果是macOS系统，设备端口可能类似于/dev/cu.usbserial-1110 '''
        self.__serialHandle = serial.Serial(port, rate)  # 连接串口， 机械手掌默认波特率为9600
    
    def setMotionTime(self, motionTime=1024):
        '''设置机械手掌电机运动时间，单位毫秒，建议1024左右，值越小越快运动到位'''
        self.__motionTimeByte0 = int(motionTime % 256)
        self.__motionTimeByte1 = int(motionTime / 256)

    def __bending2angle(self, bending, thumb=False):
        angle = bending * 2000 + 500

        if thumb == True: # for thumb
            angle = (1 - bending) * 2000 + 500 #大拇指电机相反

        if angle < 500:
            angle = 500
        if angle > 2500:
            angle = 2500

        byte0 = int(angle % 256)
        byte1 = int(angle / 256)
        return byte0, byte1

    def setFinger1(self, bending):
        '''手指1（大拇指）弯曲程度，bending=0为完全弯曲，1为完全伸直'''
        byte0, byte1 = self.__bending2angle(bending, True)
        command = [0x55,0x55, 0x08, 0x03, 0x01, self.__motionTimeByte0, self.__motionTimeByte1, 0x01, byte0, byte1]
        self.__serialHandle.write(serial.to_bytes(command))

    def setFinger2(self, bending):
        '''手指2（食指）弯曲程度，bending=0为完全弯曲，1为完全伸直'''
        byte0, byte1 = self.__bending2angle(bending)
        command = [0x55,0x55, 0x08, 0x03, 0x01, self.__motionTimeByte0, self.__motionTimeByte1, 0x02, byte0, byte1]
        self.__serialHandle.write(serial.to_bytes(command))

    def setFinger3(self, bending):
        '''手指3（中指）弯曲程度，bending=0为完全弯曲，1为完全伸直'''
        byte0, byte1 = self.__bending2angle(bending)
        command = [0x55,0x55, 0x08, 0x03, 0x01, self.__motionTimeByte0, self.__motionTimeByte1, 0x03, byte0, byte1]
        self.__serialHandle.write(serial.to_bytes(command))

    def setFinger4(self, bending):
        '''手指4（无名指）弯曲程度，bending=0为完全弯曲，1为完全伸直'''
        byte0, byte1 = self.__bending2angle(bending)
        command = [0x55,0x55, 0x08, 0x03, 0x01, self.__motionTimeByte0, self.__motionTimeByte1, 0x04, byte0, byte1]
        self.__serialHandle.write(serial.to_bytes(command))

    def setFinger5(self, bending):
        '''手指5（小指）弯曲程度，bending=0为完全弯曲，1为完全伸直'''
        byte0, byte1 = self.__bending2angle(bending)
        command = [0x55,0x55, 0x08, 0x03, 0x01, self.__motionTimeByte0, self.__motionTimeByte1, 0x05, byte0, byte1]
        self.__serialHandle.write(serial.to_bytes(command))

    def setWrist(self, bending):
        '''手腕位置，bending=0转到最左侧，1转到最右侧'''
        byte0, byte1 = self.__bending2angle(bending)
        command = [0x55,0x55, 0x08, 0x03, 0x01, self.__motionTimeByte0, self.__motionTimeByte1, 0x06, byte0, byte1]
        self.__serialHandle.write(serial.to_bytes(command))

    def setFingers(self, bending1, bending2, bending3, bending4, bending5):
        '''一次性设置五个手指的弯曲程度，bending=0为完全弯曲，1为完全伸直'''
        b10, b11 = self.__bending2angle(bending1, True)
        b20, b21 = self.__bending2angle(bending2)
        b30, b31 = self.__bending2angle(bending3)
        b40, b41 = self.__bending2angle(bending4)
        b50, b51 = self.__bending2angle(bending5)
        command = [0x55,0x55,0x14,0x03,0x05, self.__motionTimeByte0, self.__motionTimeByte1, 0x01,b10,b11, 0x02,b20,b21, 0x03,b30,b31, 0x04,b40,b41, 0x05,b50, b51]
        self.__serialHandle.write(serial.to_bytes(command))

    def setScissors(self):
        '''剪刀，剪刀石头布游戏手势之一'''
        self.setFingers(0, 1, 1, 0, 0)        
    
    def setRock(self):
        '''石头，剪刀石头布游戏手势之一'''
        self.setFingers(0, 0, 0, 0, 0)

    def setPaper(self):
        '''布，剪刀石头布游戏手势之一'''
        self.setFingers(1, 1, 1, 1, 1)

    def setNoAction(self):
        '''未出手姿势：剪刀石头布游戏手势之一'''
        self.setFingers(0, 1, 1, 1, 1)
