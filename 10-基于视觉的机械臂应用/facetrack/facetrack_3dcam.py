#!/usr/bin/env python3
# encoding:utf-8

import time
import cv2 as cv
import numpy as np
from RobotArm import RobotArm

def detect_face(detector, image):
    ''' 人脸检测函数
    '''
    h, w, c = image.shape
    if detector.getInputSize() != (w, h):
        detector.setInputSize((w, h))

    faces = detector.detect(image)
    return [] if faces[1] is None else faces[1]


if __name__ == '__main__':
    # 连接机械臂
    arm = RobotArm("/dev/ttyUSB0", 115200)

    # 初始化机械臂姿势，让机械臂呈站立姿态；顶部转90度，让摄像头平视
    arm.setAngle(1, -90, 1000) # 电机1的角度为-90（张开爪子），1000ms完成调整
    time.sleep(1.0) # 暂停1.0秒，等待电机调整完毕
    arm.setAngle(2, 0, 1000)
    time.sleep(1.0) # 暂停1.0秒，等待电机调整完毕
    arm.setAngle(3, -90, 1000) # 电机3转到-90度，让摄像头平视
    time.sleep(1.0) # 暂停1.0秒，等待电机调整完毕
    arm.setAngle(4, 0, 1000)
    time.sleep(1.0) # 暂停1.0秒，等待电机调整完毕
    arm.setAngle(5, 0, 1000) 
    time.sleep(1.0) # 暂停1.0秒，等待电机调整完毕
    arm.setAngle(6, 0, 1000) 
    time.sleep(1.0) # 暂停1.0秒，等待电机调整完毕

    # 打开摄像头，如果失败，修改device_id为0，1，2中的某个值，继续尝试
    # 仍然失败，检查USB摄像头是否连接
    orbbec_cap = cv.VideoCapture(device_id, cv.CAP_OBSENSOR)
    if orbbec_cap.isOpened() == False:
        print("Fail to open obsensor capture.")
        exit(0)    
    # 设置图像为320x240大小
    # 因为320x240检测效果最好，图像太大容易出误检测
    w = 320
    h = 240
    #创建一个窗口，用于显示图像
    cv.namedWindow("Camera", 0)

    # 装载人脸检测ONNX模型
    detector = cv.FaceDetectorYN.create(
        "face_detection_yunet_2022mar.onnx",
        "",
        (h, w),
        score_threshold=0.99, #阈值，应<1，越大误检测越少
        backend_id=cv.dnn.DNN_BACKEND_TIMVX, #使用TIMVX后端，如果不使用NPU加速，而使用CPU计算，注释掉此行及下一行
        target_id=cv.dnn.DNN_TARGET_NPU #使用NPU
    )

    
    fps_list = []
    tm = cv.TickMeter()

    # 循环，碰到按键盘就退出
    while cv.waitKey(1) < 0:
        # 从相机获取帧数据
        if orbbec_cap.grab():
            # 解码grab()获取的帧数据
            # rgb数据
            ret_bgr, frame = orbbec_cap.retrieve(None, cv.CAP_OBSENSOR_BGR_IMAGE)
            if ret_bgr:
                tm.start()
                # 检测人脸
                faces = detect_face(detector, frame) 
                tm.stop()
                # 把FPS数值放到一个列表中
                fps_list.append(tm.getFPS())
                tm.reset()
                # 列表最长为50，超过则删除首个
                if len(fps_list) > 50:
                    del fps_list[0]
                # 这样计算出最近50帧的平均FPS
                mean_fps = np.mean(fps_list)

                # 把FPS速度画到图像左上角
                cv.putText(frame, 'FPS:{:.2f}'.format(mean_fps), (0, 15), cv.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0))

                for face in faces:
                    # 把人脸框画到图像上
                    bbox = face[0:4].astype(np.int32)
                    cv.rectangle(frame, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 255, 0), 2)

                    # 计算人脸框的中心
                    centerx = bbox[0] + bbox[2] / 2  
                    centery = bbox[1] + bbox[3] / 2 
                    # 为了让人脸处于中心，摄像头在x和y方向应该移动的角度
                    stepx = (centerx - w /2) / (-15)
                    stepy = (centery - h /2) / (-15)
                    # 当前电机6和电机3的角度
                    oldanglex = arm.getAngle(6)
                    oldangley = arm.getAngle(3)
                    # 转动电机6和电机3
                    arm.setAngle(6, oldanglex + stepx, 100)
                    arm.setAngle(3, oldangley + stepy, 100)
                    time.sleep(0.1)
                    # 停止循环，只根据第一个人脸移动机械臂，忽略其他人脸
                    break 

                # 把结果图像显示到窗口里
                cv.imshow("Camera", frame)
        
    arm.unloadBusServo(1) #马达卸载动力
    arm.unloadBusServo(2) #马达卸载动力
    arm.unloadBusServo(3) #马达卸载动力
    arm.unloadBusServo(4) #马达卸载动力
    arm.unloadBusServo(5) #马达卸载动力
    arm.unloadBusServo(6) #马达卸载动力


