#!/usr/bin/env python3
# encoding:utf-8

import numpy as np 
import cv2 as cv


def main():
    # 打开深度相机
    orbbec_cap = cv.VideoCapture(0, cv.CAP_OBSENSOR)
    if orbbec_cap.isOpened() == False:
        print("Fail to open obsensor capture!")
        exit(0)

    # 装载人脸检测ONNX模型
    detector = cv.FaceDetectorYN.create(
        "face_detection_yunet_2022mar.onnx",
        "",
        (640, 480), # 设置检测器处理的图像大小wh
        score_threshold=0.99, # 阈值，应<1，越大误检测越少
        # backend_id=cv.dnn.DNN_BACKEND_TIMVX, # 使用TIMVX后端，如果不适用NPU加速，而使用CPU计算，注释掉此行及下一行
        # target_id=cv.dnn.DNN_TARGET_NPU # 使用NPU
    )

    while cv.waitKey(1) < 0:
        # 从相机获取帧数据
        if orbbec_cap.grab():
            # 解码grab()获取的帧数据
            # rgb数据
            ret_bgr, bgr_image = orbbec_cap.retrieve(None, cv.CAP_OBSENSOR_BGR_IMAGE)             
            # 深度数据
            ret_depth, depth_map = orbbec_cap.retrieve(None, cv.CAP_OBSENSOR_DEPTH_MAP)

            if ret_bgr and ret_depth:
                # 检测人脸
                faces = detector.detect(bgr_image)
                # 判断是否是真人的脸
                flags = anti_spoofing(faces, depth_map)
                visualize(faces, flags, bgr_image)
                cv.imwrite("anti-spoofing.jpg", bgr_image)
                cv.imshow("Demo", bgr_image)
        else:
            print("Fail to grab data from camera!")

    orbbec_cap.release()
          

def anti_spoofing(faces, depth_map):
    face_flags = []

    if faces[1] is not None:
        thickness = 2
        #每一张脸
        for idx, face in enumerate(faces[1]):
            # 人脸框和5点关键点数据
            coords = face[:-1].astype(np.int32)

            std = -1
            dist = []
            h, w = depth_map.shape
            # 5个点关键点到相机的距离（单位mm）
            for i in range(2, 7):
                if 0 <= coords[2*i] < w and 0 <= coords[2*i+1] < h:
                    dist.append(depth_map[coords[2*i+1], coords[2*i]])
            if len(dist) == 5: # 计算全部5个关键点到相机的距离的均方差
                std = np.std(dist)
            print("std: {:.2f}".format(std))
        
            # 使用限制条件：人脸正对相机
            if std < 0: # 未判断
                face_flags.append(0)
            elif std < 5.: # 伪造
                face_flags.append(-1)
            else: # 真人
                face_flags.append(1)

    return face_flags

def visualize(faces, face_flags, bgr_image):
    color = [(0, 0, 255), (255, 0, 0), (0, 255, 0)]

    if faces[1] is not None:
        thickness = 2
        #每一张脸
        for idx, face in enumerate(faces[1]):
            # 人脸框和5点关键点数据
            coords = face[:-1].astype(np.int32)

            # 绘制人脸框
            # 蓝色-未判断；红色-伪造；绿色-真人
            cv.rectangle(bgr_image, (coords[0], coords[1]), (coords[0]+coords[2], coords[1]+coords[3]), color[face_flags[idx]+1], thickness)

            # 绘制5个关键点
            cv.circle(bgr_image, (coords[4], coords[5]), 2, (255, 0, 0), thickness)
            cv.circle(bgr_image, (coords[6], coords[7]), 2, (0, 0, 255), thickness)
            cv.circle(bgr_image, (coords[8], coords[9]), 2, (0, 255, 0), thickness)
            cv.circle(bgr_image, (coords[10], coords[11]), 2, (255, 0, 255), thickness)
            cv.circle(bgr_image, (coords[12], coords[13]), 2, (0, 255, 255), thickness)


if __name__ == '__main__':
    main()