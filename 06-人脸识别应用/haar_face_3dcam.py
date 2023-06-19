#!/usr/bin/env python3
# encoding:utf-8


import sys
import cv2 as cv


def detectAndDisplay(frame, classifier):
    #转为灰度图
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # 人脸检测
    faces = classifier.detectMultiScale(frame_gray)
    for (x,y,w,h) in faces:
        # 绘制人脸矩形框
        frame = cv.rectangle(frame, (x, y, w, h), (0, 255, 0), 3)
        
    cv.imshow('Haar face', frame)


def main():
    # 创建分类器对象
    face_cascade = cv.CascadeClassifier()

    # 读入分类器文件
    if not face_cascade.load('haarcascade_frontalface_alt.xml'):
        print('--(!)Error loading face cascade')
        sys.exit(1)

    # 打开深度相机
    orbbec_cap = cv.VideoCapture(0, cv.CAP_OBSENSOR)
    if orbbec_cap.isOpened() == False:
        print('--(!)Error opening video capture')
        sys.exit(1)
    while cv.waitKey(1) < 0:
        # 从相机获取帧数据
        if orbbec_cap.grab():
            # 解码grab()获取的帧数据
            # rgb数据
            ret_bgr, frame = orbbec_cap.retrieve(None, cv.CAP_OBSENSOR_BGR_IMAGE)
            if ret_bgr:
                # 进行人脸检测并显示结果
                detectAndDisplay(frame, face_cascade)

    orbbec_cap.release()
    cv.destroyAllWindows()    


if __name__ == '__main__':
    main()
