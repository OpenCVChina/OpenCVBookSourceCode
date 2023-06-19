#!/usr/bin/env python3
# encoding:utf-8

import sys
import numpy as np
import cv2 as cv
import argparse

parser = argparse.ArgumentParser(description='camshift算法演示。视频文件可以从: \
                                              https://www.bogotobogo.com/python/OpenCV_Python/images/mean_shift_tracking/slow_traffic_small.mp4下载')
parser.add_argument('--video', type=str, default='slow_traffic_small.mp4', help='视频文件路径')
args = parser.parse_args()

cap = cv.VideoCapture(args.video)
if cap.isOpened() == False:
    print('打开视频文件失败。')
    sys.exit(1)
# 读入视频第一帧
ret,frame = cap.read()
# 手动设置目标初始位置
x, y, w, h = 300, 200, 100, 50 
track_window = (x, y, w, h)
# 设定要跟踪的ROI区域
roi = frame[y:y+h, x:x+w]
hsv_roi =  cv.cvtColor(roi, cv.COLOR_BGR2HSV)
mask = cv.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
roi_hist = cv.calcHist([hsv_roi],[0],mask,[180],[0,180])
cv.normalize(roi_hist,roi_hist,0,255,cv.NORM_MINMAX)
# 设定终止条件，10次迭代或移动1个像素以上
term_crit = ( cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1 )
while(1):
    ret, frame = cap.read()
    if ret == True:
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        dst = cv.calcBackProject([hsv],[0],roi_hist,[0,180],1)
        # 对新位置应用camshift
        ret, track_window = cv.CamShift(dst, track_window, term_crit)
        # 将结果绘制在图像上
        pts = cv.boxPoints(ret)
        pts = np.int0(pts)
        img2 = cv.polylines(frame,[pts],True, 255,2)
        cv.imshow('Camshift',img2)
        k = cv.waitKey(30) & 0xff
        if k == 27:
            break
    else:
        break
