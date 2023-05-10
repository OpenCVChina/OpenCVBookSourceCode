#!/usr/bin/env python3
# encoding:utf-8

import time
import cv2 as cv
import numpy as np

if __name__ == '__main__':
    # 打开摄像头，如果失败，修改参数为0，1，2中的某个值，继续尝试
    # 仍然失败，检查USB摄像头是否连接
    cap = cv.VideoCapture(0)
    # 设置采集图像为320x240大小
    # 因为320x240检测效果最好，图像太大容易出误检测
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 240)
    w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    # 打印尺寸，用于确认
    print("Image size: ", w, "x", h)
    #创建一个窗口，用于显示图像
    cv.namedWindow("Camera", 0)

    # 装载人脸检测ONNX模型
    detector = cv.FaceDetectorYN.create(
        "face_detection_yunet_2022mar.onnx",
        "",
        (w, h), # 设置检测器处理的图像大小
        score_threshold=0.99, #阈值，应<1，越大误检测越少
        backend_id=cv.dnn.DNN_BACKEND_TIMVX, #使用TIMVX后端，如果不适用NPU加速，而使用CPU计算，注释掉此行及下一行
        target_id=cv.dnn.DNN_TARGET_NPU #使用NPU
    )

    fps_list = []
    tm = cv.TickMeter()

    # 循环，碰到按键盘就退出
    while cv.waitKey(1) < 0:
        #读一帧图像
        hasFrame, frame = cap.read()
        if not hasFrame: #如果读数据失败
            print('No frames grabbed!')
            break

        #计时开始，用于计算FPS
        tm.start()
        # 检测人脸
        faces = detector.detect(frame)
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
        #把检测结果绘制到图像上
        if faces[1] is not None:
            thickness = 2
            #每一个脸
            for idx, face in enumerate(faces[1]):
                coords = face[:-1].astype(np.int32)
                # 把人脸框画到图像上
                cv.rectangle(frame, (coords[0], coords[1]), (coords[0]+coords[2], coords[1]+coords[3]), (0, 255, 0), thickness)
                # # 人脸分数
                # cv.putText(frame, '{:.2f}'.format( face[-1]), (coords[0], coords[1]), cv.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0))
                # 五个关键点
                cv.circle(frame, (coords[4], coords[5]), 2, (255, 0, 0), thickness)
                cv.circle(frame, (coords[6], coords[7]), 2, (0, 0, 255), thickness)
                cv.circle(frame, (coords[8], coords[9]), 2, (0, 255, 0), thickness)
                cv.circle(frame, (coords[10], coords[11]), 2, (255, 0, 255), thickness)
                cv.circle(frame, (coords[12], coords[13]), 2, (0, 255, 255), thickness)

        # 把结果图像显示到窗口里
        cv.imshow("Camera", frame)

