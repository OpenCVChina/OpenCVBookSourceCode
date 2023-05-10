#!/usr/bin/env python3
# encoding:utf-8


# https://github.com/opencv/opencv_contrib/blob/4.x/modules/wechat_qrcode/samples/qrcode.py

import cv2
import sys
import numpy as np

print('微信QR码识别演示:')
camIdx = 0

try:
    detector = cv2.wechat_qrcode_WeChatQRCode(
        "detect.prototxt", "detect.caffemodel", "sr.prototxt", "sr.caffemodel")
except:
    print("---------------------------------------------------------------")
    print("初始化WeChatQRCode失败.")
    print("请下载'detector.*'和'sr.*'")
    print("下载地址 https://github.com/WeChatCV/opencv_3rdparty/tree/wechat_qrcode")
    print("并将其放在当前目录下.")
    print("---------------------------------------------------------------")
    exit(0)

prevstr = ""

cap = cv2.VideoCapture(camIdx)
while True:
    res, img = cap.read()
    if img is None:
        break
    res, points = detector.detectAndDecode(img)

    for idx in range(len(points)):
        box = points[idx].astype(np.int32)
        cv2.drawContours(img, [box], -1, (0,255,0), 3)
        print('QR code{}: {}'.format(idx, res[idx]))

    # for t in res:
    #     if t != prevstr:
    #         print(t)
    # if res:
    #     prevstr = res[-1]
    cv2.imshow("QRCode", img)
    if cv2.waitKey(30) >= 0:
        break

cap.release()
cv2.destroyAllWindows()
