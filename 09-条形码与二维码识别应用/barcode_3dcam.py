#!/usr/bin/env python3
# encoding:utf-8


import cv2 as cv
import numpy as np


def main():

    device_id = 0
    orbbec_cap = cv.VideoCapture(device_id, cv.CAP_OBSENSOR)
    if not orbbec_cap.isOpened():
        print('Failed to open camera.')
        exit(0)

    sr_prototxt = 'sr.prototxt'
    sr_model = 'sr.caffemodel'
    bar_det = cv.barcode.BarcodeDetector(sr_prototxt, sr_model)

    while cv.waitKey(1) < 0:
        # 从相机获取帧数据
        if orbbec_cap.grab():
            # print("Grabbing data succeeds.")

            # 解码grab()获取的帧数据
            # rgb数据
            ret_bgr, frame = orbbec_cap.retrieve(None, cv.CAP_OBSENSOR_BGR_IMAGE)
            if ret_bgr:
                grey_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

                # 条形码检测和解码
                retval, decode_info, decode_type, corners = bar_det.detectAndDecodeWithType(grey_frame)

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

    orbbec_cap.release()
    cv.destroyAllWindows() 



if __name__ == '__main__':
    main()
