#!/usr/bin/env python3
# encoding:utf-8


import sys
import numpy as np
import cv2 as cv

def main():
    # 设置视频的宽度和高度
    frame_size = (320, 240)

    # 设置帧率
    fps = 25

    # 视频编解码格式
    fourcc = cv.VideoWriter_fourcc('M','J','P','G')

    # 创建writer
    writer = cv.VideoWriter("myvideo.avi", fourcc, fps, frame_size)
    # 检查是否创建成功
    if writer.isOpened() == False:
        print("Error creating video writer.")
        sys.exit()

    for i in range(0, 100):

        # 设置视频帧画面
        im = np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8)

        # 将数字绘制到画面上
        cv.putText(im, str(i), (int(frame_size[0]/3), int(frame_size[1]*2/3)), cv.FONT_HERSHEY_SIMPLEX, 3.0, (255, 255, 255), 3)

        # 保存视频帧到文件"myvideo.avi"
        writer.write(im)

    # 释放writer
    writer.release()


if __name__  == '__main__':
    main()
