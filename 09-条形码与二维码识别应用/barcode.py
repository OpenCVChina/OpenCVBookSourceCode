#!/usr/bin/env python3
# encoding:utf-8


import cv2 as cv
import numpy as np


def main():

    device_id = 0
    cap = cv.VideoCapture(device_id)
    if not cap.isOpened():
        print('Failed to open camera.')
        exit(0)

    sr_prototxt = 'sr.prototxt'
    sr_model = 'sr.caffemodel'
    bar_det = cv.barcode.BarcodeDetector(sr_prototxt, sr_model)

    while cv.waitKey(1) < 0:
        hasFrame, frame = cap.read()
        if not hasFrame:
            print('Failed to grab frame.')
            break

        grey_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # 条形码检测和解码
        retval, decode_info, decode_type, corners = bar_det.detectAndDecode(grey_frame)

        # 绘制结果
        if corners is not None:
            # 条形码位置
            corners = corners.astype(np.int32)
            cv.drawContours(frame, corners, -1, (0,255,0), 3)

            # 条形码内容
            if decode_info is not None:
                for idx in range(corners.shape[0]):
                    if len(decode_info) > idx:
                        print('Barcode{}, Type: {}, Info: {}'.format(idx, decode_type[idx], decode_info[idx]))
                    else:
                        print('Failed to decode barcode {}.'.format(idx))
            else:
                print('Failed to decode.')

        cv.imshow('Barcode', frame)  

    cap.release()
    cv.destroyAllWindows() 



if __name__ == '__main__':
    main()
