#!/usr/bin/env python3
# encoding:utf-8


import sys
import cv2 as cv


def main():
    # 打开深度相机
    orbbec_cap = cv.VideoCapture(0, cv.CAP_OBSENSOR)

    # 检查是否打开成功
    if orbbec_cap.isOpened() == False:
        print("Fail to open obsensor capture.")
        sys.exit

    while True:
        # 从相机获取帧数据
        if orbbec_cap.grab():
            # rgb数据
            ret_bgr, bgr_image = orbbec_cap.retrieve(None, cv.CAP_OBSENSOR_BGR_IMAGE)
            if ret_bgr:
                # 显示视频帧
                cv.imshow("BGR", bgr_image)
        else:
            print("Fail to grab data from camera.")

        if cv.waitKey(30) >= 0:
            break

    # 销毁窗口
    cv.destroyAllWindows()
    # 释放orbbec_cap
    orbbec_cap.release()

if __name__ == '__main__':
    main()
