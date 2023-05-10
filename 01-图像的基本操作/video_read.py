#!/usr/bin/env python3
# encoding:utf-8


import sys
import cv2 as cv


def main():
    # 打开第一个摄像头
    cap = cv.VideoCapture(0)
    # 打开视频文件
    #cap = cv.VideoCapture('slow_traffic_small.mp4')

    # 检查是否打开成功
    if cap.isOpened() == False:
        print('Error opening the video source.')
        sys.exit()

    while True:
        # 读取一帧视频，存放到im
        ret, im = cap.read()
        if not ret:
            print('No image read.')
            break

        # 显示视频帧
        cv.imshow('Live', im)
        # 等待30毫秒，如果有按键则退出循环
        if cv.waitKey(30) >= 0:
            break

    # 销毁窗口
    cv.destroyAllWindows()
    # 释放cap
    cap.release()

if __name__ == '__main__':
    main()
