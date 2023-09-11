#!/usr/bin/env python3
# encoding:utf-8
import sys
import time
import cv2 as cv
import numpy as np
from RobotArm import RobotArm

def find_largest_face(faces):
    facearea = 0
    lface = []
    for face in faces:
        bboxarea = face[2] * face[3]
        if bboxarea > facearea:
            facearea = bboxarea
            lface = face
    return None if facearea == 0 else lface

def armStand(arm):
    # 初始化机械臂姿势，让机械臂呈站立姿态；顶部转90度，让摄像头平视
    movetime = 1000 # 毫秒
    arm.setAngle(1, -90) # 电机1的角度为-90（张开爪子）
    time.sleep(movetime/1000) # 暂停X秒，等待电机调整完毕
    arm.setAngle(2, 0)
    time.sleep(movetime/1000)
    arm.setAngle(3, -70) # 电机3转到-70度，让摄像头略向上仰视
    time.sleep(movetime/1000)
    arm.setAngle(4, 0)
    time.sleep(movetime/1000)
    arm.setAngle(5, 0, 2000) 
    time.sleep(movetime/1000)
    arm.setAngle(6, 0) # 电机6的角度为0
    time.sleep(movetime/1000)  

def rotateArm(arm, face):
    # 计算人脸框的中心
    centerx = face[0] + face[2] / 2  
    centery = face[1] + face[3] / 2 
    # 为了让人脸处于中心，摄像头在x和y方向应该移动的角度
    stepx = (centerx - w /2) / (-50)
    stepy = (centery - h /2) / (-50)
    # 当前电机6和电机3的角度
    oldanglex = arm.getAngle(6)
    oldangley = arm.getAngle(3)
    # 转动电机6和电机3
    arm.setAngle(6, oldanglex + stepx)
    arm.setAngle(3, oldangley + stepy)
    time.sleep(0.01)


if __name__ == '__main__':
    # 连接机械臂
    arm = RobotArm("/dev/ttyUSB0", 115200)
    # 让机械臂站立到初始位置
    armStand(arm)

    # 打开摄像头，如果失败，修改device_id为0，1，2中的某个值，继续尝试
    # 仍然失败，检查USB摄像头是否连接
    device_id = 0
    cap = cv.VideoCapture(device_id)
    if not cap.isOpened():
        print("Failed to open the camera.")
        sys.exit(1)

    # 设置采集图像为640x480大小
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
    w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    # 打印尺寸，用于确认
    print("Image size: ", w, "x", h)

    #创建一个窗口，用于显示图像
    cv.namedWindow("Camera", cv.WINDOW_AUTOSIZE)

    # 装载人脸检测ONNX模型
    detector = cv.FaceDetectorYN.create(
        "face_detection_yunet_2023mar.onnx",
        "",
        (320, 240), #输入图像尺寸
        score_threshold=0.9, #阈值，应<1，越大误检测越少
        #backend_id=cv.dnn.DNN_BACKEND_TIMVX, #使用TIMVX后端，如果不适用NPU加速，而使用CPU计算，注释掉此行及下一行
        #target_id=cv.dnn.DNN_TARGET_NPU #使用NPU
    )
    # 装载人脸识别ONNX模型
    recognizer = cv.FaceRecognizerSF.create(
        "face_recognition_sface_2021dec.onnx",
        "",
        #backend_id=cv.dnn.DNN_BACKEND_TIMVX, #使用TIMVX后端，如果不适用NPU加速，而使用CPU计算，注释掉此行及下一行
        #target_id=cv.dnn.DNN_TARGET_NPU #使用NPU
    )
    #一些全局参数
    mode = "detect"
    imageScale = w / 320
    ownerFaceFeature = []
    missedFrames = 0
    l2_threshold = 1.128

    # 循环，碰到按键盘就退出
    while cv.waitKey(1) < 0:
        #读一帧图像
        hasFrame, frame = cap.read()
        if not hasFrame:
            print('No frames grabbed!')
            break

        # 在320宽度的图像上检测人脸效果最佳
        frame320 = cv.resize(frame, (320, 240))
        retval, faces = detector.detect(frame320)
        if faces is None:
            faces = []
        else:
            # 将检测结果乘以系数，使之为原始大图像的人脸位置
            faces *= imageScale

        if mode == "detect":
            lface = find_largest_face(faces)
            if lface is not None:
                aligned_face = recognizer.alignCrop(frame, lface)
                ownerFaceFeature = recognizer.feature(aligned_face)
                mode = "tracking"
                print("Swith to tracking mode")
    
        elif mode == "tracking":
            missedFrames += 1

            for face in faces:
                aligned_face = recognizer.alignCrop(frame, face) #获取对齐后的人脸
                afeature = recognizer.feature(aligned_face) #提取人脸特征
                score = recognizer.match(ownerFaceFeature, afeature, cv.FaceRecognizerSF_FR_NORM_L2) #比较人脸相似分数，越低越相似
                # print("score=", score)

                # 把人脸框以蓝色画到图像上
                bbox = face[0:4].astype(np.int32)
                cv.rectangle(frame, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (255, 0, 0), 2)

                if score < l2_threshold: #匹配成功
                    missedFrames = 0
                    # 把认识的人脸，以绿色画到图像上
                    cv.rectangle(frame, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 255, 0), 3)
                    rotateArm(arm, face) #转动机械臂对准
                    break 

            if missedFrames >= 100: #如果100帧还找不到以前的人脸
                missedFrames = 0
                mode = "detect" #重新查找一个新人脸
                print("Swith to detection mode")
        else:
            print("Error mode code.")
            break

        # 把结果图像显示到窗口里
        cv.imshow("Camera", frame)

    #退出程序前，将所有马达卸力
    arm.unloadBusServo(1)
    arm.unloadBusServo(2)
    arm.unloadBusServo(3)
    arm.unloadBusServo(4)
    arm.unloadBusServo(5)
    arm.unloadBusServo(6)
    
