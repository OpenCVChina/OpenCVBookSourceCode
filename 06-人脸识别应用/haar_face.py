#!/usr/bin/env python3
# encoding:utf-8


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
        exit(0)

    # 打开摄像头，获取视频帧
    cap = cv.VideoCapture(0)
    if not cap.isOpened:
        print('--(!)Error opening video capture')
        exit(0)
    #frame = cv.imread('gemma.jpeg')
    while cv.waitKey(1) < 0:
        ret, frame = cap.read()
        if frame is None:
            print('--(!) No captured frame -- Break!')
            break
        # 进行人脸检测并显示结果
        detectAndDisplay(frame, face_cascade)


if __name__ == '__main__':
    main()
