#!/usr/bin/env python3
# encoding:utf-8

import serial
import ctypes
import time
import logging
import numpy as np
from math import *

#幻尔科技总线舵机通信#
LOBOT_SERVO_FRAME_HEADER         = 0x55
LOBOT_SERVO_MOVE_TIME_WRITE      = 1
LOBOT_SERVO_MOVE_TIME_READ       = 2
LOBOT_SERVO_MOVE_TIME_WAIT_WRITE = 7
LOBOT_SERVO_MOVE_TIME_WAIT_READ  = 8
LOBOT_SERVO_MOVE_START           = 11
LOBOT_SERVO_MOVE_STOP            = 12
LOBOT_SERVO_ID_WRITE             = 13
LOBOT_SERVO_ID_READ              = 14
LOBOT_SERVO_ANGLE_OFFSET_ADJUST  = 17
LOBOT_SERVO_ANGLE_OFFSET_WRITE   = 18
LOBOT_SERVO_ANGLE_OFFSET_READ    = 19
LOBOT_SERVO_ANGLE_LIMIT_WRITE    = 20
LOBOT_SERVO_ANGLE_LIMIT_READ     = 21
LOBOT_SERVO_VIN_LIMIT_WRITE      = 22
LOBOT_SERVO_VIN_LIMIT_READ       = 23
LOBOT_SERVO_TEMP_MAX_LIMIT_WRITE = 24
LOBOT_SERVO_TEMP_MAX_LIMIT_READ  = 25
LOBOT_SERVO_TEMP_READ            = 26
LOBOT_SERVO_VIN_READ             = 27
LOBOT_SERVO_POS_READ             = 28
LOBOT_SERVO_OR_MOTOR_MODE_WRITE  = 29
LOBOT_SERVO_OR_MOTOR_MODE_READ   = 30
LOBOT_SERVO_LOAD_OR_UNLOAD_WRITE = 31
LOBOT_SERVO_LOAD_OR_UNLOAD_READ  = 32
LOBOT_SERVO_LED_CTRL_WRITE       = 33
LOBOT_SERVO_LED_CTRL_READ        = 34
LOBOT_SERVO_LED_ERROR_WRITE      = 35
LOBOT_SERVO_LED_ERROR_READ       = 36

# CRITICAL, ERROR, WARNING, INFO, DEBUG
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

# 4自由度机械臂逆运动学：给定相应的坐标（X,Y,Z），以及俯仰角，计算出每个关节转动的角度
# 2020/07/20 Aiden
class IK:
    # 舵机从下往上数
    # 公用参数，即4自由度机械臂的连杆参数
    l1 = 6.10    #机械臂底盘中心到第二个舵机中心轴的距离6.10cm
    l2 = 10.16   #第二个舵机到第三个舵机的距离10.16cm
    l3 = 9.64    #第三个舵机到第四个舵机的距离9.64cm
    l4 = 0.00    #这里不做具体赋值，根据初始化时的选择进行重新赋值

    # 气泵款特有参数
    l5 = 4.70  #第四个舵机到吸嘴正上方的距离4.70cm
    l6 = 4.46  #吸嘴正上方到吸嘴的距离4.46cm
    alpha = degrees(atan(l6 / l5))  #计算l5和l4的夹角

    def __init__(self, arm_type): #根据不同款的夹持器，适配参数
        self.arm_type = arm_type
        if self.arm_type == 'pump': #如果是气泵款机械臂
            self.l4 = sqrt(pow(self.l5, 2) + pow(self.l6, 2))  #第四个舵机到吸嘴作为第四个连杆
        elif self.arm_type == 'arm':
            self.l4 = 16.65  #第四个舵机到机械臂末端的距离16.6cm， 机械臂末端是指爪子完全闭合时

    def setLinkLength(self, L1=l1, L2=l2, L3=l3, L4=l4, L5=l5, L6=l6):
        # 更改机械臂的连杆长度，为了适配相同结构不同长度的机械臂
        self.l1 = L1
        self.l2 = L2
        self.l3 = L3
        self.l4 = L4
        self.l5 = L5
        self.l6 = L6
        if self.arm_type == 'pump':
            self.l4 = sqrt(pow(self.l5, 2) + pow(self.l6, 2))
            self.alpha = degrees(atan(self.l6 / self.l5))

    def getLinkLength(self):
        # 获取当前设置的连杆长度
        if self.arm_type == 'pump':
            return {"L1":self.l1, "L2":self.l2, "L3":self.l3, "L4":self.l4, "L5":self.l5, "L6":self.l6}
        else:
            return {"L1":self.l1, "L2":self.l2, "L3":self.l3, "L4":self.l4}

    def getRotationAngle(self, coordinate_data, Alpha):
        # 给定指定坐标和俯仰角，返回每个关节应该旋转的角度，如果无解返回False
        # coordinate_data为夹持器末端坐标，坐标单位cm， 以元组形式传入，例如(0, 5, 10)
        # Alpha为夹持器与水平面的夹角，单位度

        # 设夹持器末端为P(X, Y, Z), 坐标原点为O, 原点为云台中心在地面的投影， P点在地面的投影为P_
        # l1与l2的交点为A, l2与l3的交点为B，l3与l4的交点为C
        # CD与PD垂直，CD与z轴垂直，则俯仰角Alpha为DC与PC的夹角, AE垂直DP_， 且E在DP_上， CF垂直AE，且F在AE上
        # 夹角表示：例如AB和BC的夹角表示为ABC
        X, Y, Z = coordinate_data
        if self.arm_type == 'pump':
            Alpha -= self.alpha
        #求底座旋转角度
        theta6 = degrees(atan2(Y, X))
 
        P_O = sqrt(X*X + Y*Y) #P_到原点O距离
        CD = self.l4 * cos(radians(Alpha))
        PD = self.l4 * sin(radians(Alpha)) #当俯仰角为正时，PD为正，当俯仰角为负时，PD为负
        AF = P_O - CD
        CF = Z - self.l1 - PD
        AC = sqrt(pow(AF, 2) + pow(CF, 2))
        if round(CF, 4) < -self.l1:
            logger.debug('高度低于0, CF(%s)<l1(%s)', CF, -self.l1)
            return False
        if self.l2 + self.l3 < round(AC, 4): #两边之和小于第三边
            logger.debug('不能构成连杆结构, l2(%s) + l3(%s) < AC(%s)', self.l2, self.l3, AC)
            return False

        #求theat4
        cos_ABC = round(-(pow(AC, 2)- pow(self.l2, 2) - pow(self.l3, 2))/(2*self.l2*self.l3), 4) #余弦定理
        if abs(cos_ABC) > 1:
            logger.debug('不能构成连杆结构, abs(cos_ABC(%s)) > 1', cos_ABC)
            return False
        ABC = acos(cos_ABC) #反三角算出弧度
        theta4 = 180.0 - degrees(ABC)

        #求theta5
        CAF = acos(AF / AC)
        cos_BAC = round((pow(AC, 2) + pow(self.l2, 2) - pow(self.l3, 2))/(2*self.l2*AC), 4) #余弦定理
        if abs(cos_BAC) > 1:
            logger.debug('不能构成连杆结构, abs(cos_BAC(%s)) > 1', cos_BAC)
            return False
        if CF < 0:
            zf_flag = -1
        else:
            zf_flag = 1
        theta5 = degrees(CAF * zf_flag + acos(cos_BAC))

        #求theta3
        theta3 = Alpha - theta5 + theta4
        if self.arm_type == 'pump':
            theta3 += self.alpha

        return {"theta3":theta3, "theta4":theta4, "theta5":theta5, "theta6":theta6} # 有解时返回角度字典


#机械臂根据逆运动学算出的角度进行移动
ik = IK('arm')
#设置连杆长度
l1 = ik.l1 + 0.75
l4 = ik.l4 - 0.15
ik.setLinkLength(L1=l1, L4=l4)

class RobotArm:
    serialHandle = 0
    servo3Range = (0, 1000.0, 0, 240.0) #脉宽， 角度
    servo4Range = (0, 1000.0, 0, 240.0)
    servo5Range = (0, 1000.0, 0, 240.0)
    servo6Range = (0, 1000.0, 0, 240.0)

    def __init__(self, port="/dev/ttyUSB0", rate=115200):
        self.serialHandle = serial.Serial(port, rate)  # 初始化串口， 波特率为115200
        self.setServoRange()

    def setServoRange(self, servo3_Range=servo3Range, servo4_Range=servo4Range, servo5_Range=servo5Range, servo6_Range=servo6Range):
        # 适配不同的舵机
        self.servo3Range = servo3_Range
        self.servo4Range = servo4_Range
        self.servo5Range = servo5_Range
        self.servo6Range = servo6_Range
        self.servo3Param = (self.servo3Range[1] - self.servo3Range[0]) / (self.servo3Range[3] - self.servo3Range[2])
        self.servo4Param = (self.servo4Range[1] - self.servo4Range[0]) / (self.servo4Range[3] - self.servo4Range[2])
        self.servo5Param = (self.servo5Range[1] - self.servo5Range[0]) / (self.servo5Range[3] - self.servo5Range[2])
        self.servo6Param = (self.servo6Range[1] - self.servo6Range[0]) / (self.servo6Range[3] - self.servo6Range[2])

    def transformAngelAdaptArm(self, theta3, theta4, theta5, theta6):
        #将逆运动学算出的角度转换为舵机对应的脉宽值
        servo3 = int(round(theta3 * self.servo3Param + (self.servo3Range[1] + self.servo3Range[0])/2))
        if servo3 > self.servo3Range[1] or servo3 < self.servo3Range[0] + 60:
            logger.info('servo3(%s)超出范围(%s, %s)', servo3, self.servo3Range[0] + 60, self.servo3Range[1])
            return False

        servo4 = int(round(theta4 * self.servo4Param + (self.servo4Range[1] + self.servo4Range[0])/2))
        if servo4 > self.servo4Range[1] or servo4 < self.servo4Range[0]:
            logger.info('servo4(%s)超出范围(%s, %s)', servo4, self.servo4Range[0], self.servo4Range[1])
            return False

        servo5 = int(round((self.servo5Range[1] + self.servo5Range[0])/2 - (90.0 - theta5) * self.servo5Param))
        if servo5 > ((self.servo5Range[1] + self.servo5Range[0])/2 + 90*self.servo5Param) or servo5 < ((self.servo5Range[1] + self.servo5Range[0])/2 - 90*self.servo5Param):
            logger.info('servo5(%s)超出范围(%s, %s)', servo5, self.servo5Range[0], self.servo5Range[1])
            return False
        
        if theta6 < -(self.servo6Range[3] - self.servo6Range[2])/2:
            servo6 = int(round(((self.servo6Range[3] - self.servo6Range[2])/2 + (90 + (180 + theta6))) * self.servo6Param))
        else:
            servo6 = int(round(((self.servo6Range[3] - self.servo6Range[2])/2 - (90 - theta6)) * self.servo6Param))
        if servo6 > self.servo6Range[1] or servo6 < self.servo6Range[0]:
            logger.info('servo6(%s)超出范围(%s, %s)', servo6, self.servo6Range[0], self.servo6Range[1])
            return False

        return {"servo3": servo3, "servo4": servo4, "servo5": servo5, "servo6": servo6}

    def servosMove(self, servos, movetime=None):
        #驱动3,4,5,6号舵机转动
        time.sleep(0.02)
        if movetime is None:
            max_d = 0
            for i in  range(0, 4):
                d = abs(self.getBusServoPulse(i + 3) - servos[i])
                if d > max_d:
                    max_d = d
            movetime = int(max_d*4)
        self.setBusServoPulse(3, servos[0], movetime)
        self.setBusServoPulse(4, servos[1], movetime)
        self.setBusServoPulse(5, servos[2], movetime)
        self.setBusServoPulse(6, servos[3], movetime)

        return movetime

    def setPitchRange(self, coordinate_data, alpha1, alpha2, da = 1):
        #给定坐标coordinate_data和俯仰角的范围alpha1，alpha2, 自动在范围内寻找到的合适的解
        #如果无解返回False,否则返回对应舵机角度,俯仰角
        #坐标单位cm， 以元组形式传入，例如(0, 5, 10)
        #da为俯仰角遍历时每次增加的角度
        x, y, z = coordinate_data
        if alpha1 >= alpha2:
            da = -da
        for alpha in np.arange(alpha1, alpha2, da):#遍历求解
            result = ik.getRotationAngle((x, y, z), alpha)
            if result:
                theta3, theta4, theta5, theta6 = result['theta3'], result['theta4'], result['theta5'], result['theta6']
                servos = self.transformAngelAdaptArm(theta3, theta4, theta5, theta6)
                if servos != False:
                    return servos, alpha

        return False

    def setPitchRangeMoving(self, coordinate_data, alpha, alpha1, alpha2, movetime=None):
        #给定坐标coordinate_data和俯仰角alpha,以及允许的俯仰角范围的范围alpha1, alpha2，自动寻找最接近给定俯仰角的解，并转到目标位置
        #如果无解返回False,否则返回舵机角度、俯仰角、运行时间
        #坐标单位cm， 以元组形式传入，例如(0, 5, 10)
        #alpha为给定俯仰角
        #alpha1和alpha2为俯仰角的取值范围
        #movetime为舵机转动时间，单位ms, 如果不给出时间，则自动计算
        x, y, z = coordinate_data
        result1 = self.setPitchRange((x, y, z), alpha, alpha1)
        result2 = self.setPitchRange((x, y, z), alpha, alpha2)
        if result1 != False:
            data = result1
            if result2 != False:
                if abs(result2[1] - alpha) < abs(result1[1] - alpha):
                    data = result2
        else:
            if result2 != False:
                data = result2
            else:
                return False
        servos, alpha = data[0], data[1]

        movetime = self.servosMove((servos["servo3"], servos["servo4"], servos["servo5"], servos["servo6"]), movetime)

        return servos, alpha, movetime

    def checksum(self, buf):
        # 计算校验和
        sum = 0x00
        for b in buf:  # 求和
            sum += b
        sum = sum - 0x55 - 0x55  # 去掉命令开头的两个 0x55
        sum = ~sum  # 取反
        return sum & 0xff

    def serial_serro_wirte_cmd(self, id=None, w_cmd=None, dat1=None, dat2=None):
        '''
        写指令
        :param id:
        :param w_cmd:
        :param dat1:
        :param dat2:
        :return:
        '''
        buf = bytearray(b'\x55\x55')  # 帧头
        buf.append(id)
        # 指令长度
        if dat1 is None and dat2 is None:
            buf.append(3)
        elif dat1 is not None and dat2 is None:
            buf.append(4)
        elif dat1 is not None and dat2 is not None:
            buf.append(7)

        buf.append(w_cmd)  # 指令
        # 写数据
        if dat1 is None and dat2 is None:
            pass
        elif dat1 is not None and dat2 is None:
            buf.append(dat1 & 0xff)  # 偏差
        elif dat1 is not None and dat2 is not None:
            buf.extend([(0xff & dat1), (0xff & (dat1 >> 8))])  # 分低8位 高8位 放入缓存
            buf.extend([(0xff & dat2), (0xff & (dat2 >> 8))])  # 分低8位 高8位 放入缓存
        # 校验和
        buf.append(self.checksum(buf))
        # for i in buf:
        #     print('%x' %i)
        self.serialHandle.write(buf)  # 发送

    def serial_servo_read_cmd(self, id=None, r_cmd=None):
        '''
        发送读取命令
        :param id:
        :param r_cmd:
        :param dat:
        :return:
        '''
        buf = bytearray(b'\x55\x55')  # 帧头
        buf.append(id)
        buf.append(3)  # 指令长度
        buf.append(r_cmd)  # 指令
        buf.append(self.checksum(buf))  # 校验和
        self.serialHandle.write(buf)  # 发送

    def serial_servo_get_rmsg(self, cmd):
        '''
        # 获取指定读取命令的数据
        :param cmd: 读取命令
        :return: 数据
        '''
        self.serialHandle.flushInput()  # 清空接收缓存
        time.sleep(0.005)  # 稍作延时，等待接收完毕
        count = self.serialHandle.inWaiting()    # 获取接收缓存中的字节数
        if count != 0:  # 如果接收到的数据不空
            recv_data = self.serialHandle.read(count)  # 读取接收到的数据
            # for i in recv_data:
            #     print('%#x' %ord(i))
            # 是否是读id指令
            try:
                if recv_data[0] == 0x55 and recv_data[1] == 0x55 and recv_data[4] == cmd:
                    dat_len = recv_data[3]
                    self.serialHandle.flushInput()  # 清空接收缓存
                    if dat_len == 4:
                        # print ctypes.c_int8(ord(recv_data[5])).value    # 转换成有符号整型
                        return recv_data[5]
                    elif dat_len == 5:
                        pos = 0xffff & (recv_data[5] | (0xff00 & (recv_data[6] << 8)))
                        return ctypes.c_int16(pos).value
                    elif dat_len == 7:
                        pos1 = 0xffff & (recv_data[5] | (0xff00 & (recv_data[6] << 8)))
                        pos2 = 0xffff & (recv_data[7] | (0xff00 & (recv_data[8] << 8)))
                        return ctypes.c_int16(pos1).value, ctypes.c_int16(pos2).value
                else:
                    return None
            except BaseException as e:
                print(e)
        else:
            self.serialHandle.flushInput()  # 清空接收缓存
            return None

    def setBusServoID(self, oldid, newid):
        """
        配置舵机id号, 出厂默认为1
        :param oldid: 原来的id， 出厂默认为1
        :param newid: 新的id
        """
        self.serial_serro_wirte_cmd(oldid, LOBOT_SERVO_ID_WRITE, newid)

    def getBusServoID(self, id=None):
        """
        读取串口舵机id
        :param id: 默认为空
        :return: 返回舵机id
        """
        
        while True:
            if id is None:  # 总线上只能有一个舵机
                self.serial_servo_read_cmd(0xfe, LOBOT_SERVO_ID_READ)
            else:
                self.serial_servo_read_cmd(id, LOBOT_SERVO_ID_READ)
            # 获取内容
            msg = self.serial_servo_get_rmsg(LOBOT_SERVO_ID_READ)
            if msg is not None:
                return msg

    def setBusServoPulse(self, id, pulse, use_time):
        """
        驱动串口舵机转到指定位置
        :param id: 要驱动的舵机id
        :pulse: 位置 0 - 1000 对应 0-240度
        :use_time: 转动需要的时间
        """
        pulse = 0 if pulse < 0 else pulse
        pulse = 1000 if pulse > 1000 else pulse
        use_time = 0 if use_time < 0 else use_time
        use_time = 30000 if use_time > 30000 else use_time
        self.serial_serro_wirte_cmd(id, LOBOT_SERVO_MOVE_TIME_WRITE, pulse, use_time)

    def stopBusServo(self, id=None):
        '''
        停止舵机运行
        :param id:
        :return:
        '''
        self.serial_serro_wirte_cmd(id, LOBOT_SERVO_MOVE_STOP)

    def setBusServoDeviation(self, id, d=0):
        """
        调整偏差
        :param id: 舵机id
        :param d:  偏差
        """
        self.serial_serro_wirte_cmd(id, LOBOT_SERVO_ANGLE_OFFSET_ADJUST, d)

    def saveBusServoDeviation(self, id):
        """
        配置偏差，掉电保护
        :param id: 舵机id
        """
        self.serial_serro_wirte_cmd(id, LOBOT_SERVO_ANGLE_OFFSET_WRITE)

    time_out = 50
    def getBusServoDeviation(self, id):
        '''
        读取偏差值
        :param id: 舵机号
        :return:
        '''
        # 发送读取偏差指令
        count = 0
        while True:
            self.serial_servo_read_cmd(id, LOBOT_SERVO_ANGLE_OFFSET_READ)
            # 获取
            msg = self.serial_servo_get_rmsg(LOBOT_SERVO_ANGLE_OFFSET_READ)
            count += 1
            if msg is not None:
                return msg
            if count > time_out:
                return None

    def setBusServoAngleLimit(self, id, low, high):
        '''
        设置舵机转动范围
        :param id:
        :param low:
        :param high:
        :return:
        '''
        self.serial_serro_wirte_cmd(id, LOBOT_SERVO_ANGLE_LIMIT_WRITE, low, high)

    def getBusServoAngleLimit(self, id):
        '''
        读取舵机转动范围
        :param id:
        :return: 返回元祖 0： 低位  1： 高位
        '''
        
        while True:
            self.serial_servo_read_cmd(id, LOBOT_SERVO_ANGLE_LIMIT_READ)
            msg = self.serial_servo_get_rmsg(LOBOT_SERVO_ANGLE_LIMIT_READ)
            if msg is not None:
                count = 0
                return msg

    def setBusServoVinLimit(self, id, low, high):
        '''
        设置舵机电压范围
        :param id:
        :param low:
        :param high:
        :return:
        '''
        self.serial_serro_wirte_cmd(id, LOBOT_SERVO_VIN_LIMIT_WRITE, low, high)

    def getBusServoVinLimit(self, id):
        '''
        读取舵机转动范围
        :param id:
        :return: 返回元祖 0： 低位  1： 高位
        '''
        while True:
            self.serial_servo_read_cmd(id, LOBOT_SERVO_VIN_LIMIT_READ)
            msg = self.serial_servo_get_rmsg(LOBOT_SERVO_VIN_LIMIT_READ)
            if msg is not None:
                return msg

    def setBusServoMaxTemp(self, id, m_temp):
        '''
        设置舵机最高温度报警
        :param id:
        :param m_temp:
        :return:
        '''
        self.serial_serro_wirte_cmd(id, LOBOT_SERVO_TEMP_MAX_LIMIT_WRITE, m_temp)

    def getBusServoTempLimit(self, id):
        '''
        读取舵机温度报警范围
        :param id:
        :return:
        '''
        while True:
            self.serial_servo_read_cmd(id, LOBOT_SERVO_TEMP_MAX_LIMIT_READ)
            msg = self.serial_servo_get_rmsg(LOBOT_SERVO_TEMP_MAX_LIMIT_READ)
            if msg is not None:
                return msg

    def getBusServoPulse(self, id):
        '''
        读取舵机当前位置
        :param id:
        :return:
        '''
        while True:
            self.serial_servo_read_cmd(id, LOBOT_SERVO_POS_READ)
            msg = self.serial_servo_get_rmsg(LOBOT_SERVO_POS_READ)
            if msg is not None:
                return msg


    def getBusServoTemp(self, id):
        '''
        读取舵机温度
        :param id:
        :return:
        '''
        while True:
            self.serial_servo_read_cmd(id, LOBOT_SERVO_TEMP_READ)
            msg = self.serial_servo_get_rmsg(LOBOT_SERVO_TEMP_READ)
            if msg is not None:
                return msg

    def getBusServoVin(self, id):
        '''
        读取舵机电压
        :param id:
        :return:
        '''
        while True:
            self.serial_servo_read_cmd(id, LOBOT_SERVO_VIN_READ)
            msg = self.serial_servo_get_rmsg(LOBOT_SERVO_VIN_READ)
            if msg is not None:
                return msg

    def restBusServoPulse(self, oldid):
        # 舵机清零偏差和P值中位（500）
        serial_servo_set_deviation(oldid, 0)    # 清零偏差
        time.sleep(0.1)
        self.serial_serro_wirte_cmd(oldid, LOBOT_SERVO_MOVE_TIME_WRITE, 500, 100)    # 中位

    ##马达卸载动力
    def unloadBusServo(self, id):
        self.serial_serro_wirte_cmd(id, LOBOT_SERVO_LOAD_OR_UNLOAD_WRITE, 0)


    ##检查马达动力状态
    def getBusServoLoadStatus(self, id):
        while True:
            self.serial_servo_read_cmd(id, LOBOT_SERVO_LOAD_OR_UNLOAD_READ)
            msg = self.serial_servo_get_rmsg(LOBOT_SERVO_LOAD_OR_UNLOAD_READ)
            if msg is not None:
                return msg

    def getAngle(self, id):
        """
        通过串口获取舵机位置，返回角度
        :param id: 要驱动的舵机id
        :
        """
        pulse = self.getBusServoPulse(id)
        return int(round(pulse * 240 / 1000 - 120))

    def setAngle(self, id, angle, use_time):
        """
        驱动串口舵机转到指定位置
        :param id: 要驱动的舵机id
        :angle: 0-240度 
        :use_time: 转动需要的时间
        """
        pulse = int(round((angle + 120) * 1000 / 240))
        self.setBusServoPulse(id, pulse, use_time)

