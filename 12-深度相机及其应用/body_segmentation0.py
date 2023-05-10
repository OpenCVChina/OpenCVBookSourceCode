import numpy as np 
import cv2 as cv


def main(thres):
    # 打开深度相机
    orbbec_cap = cv.VideoCapture(0, cv.CAP_OBSENSOR)
    if orbbec_cap.isOpened() == False:
        print("Fail to open obsensor capture!")
        exit(0)

    while cv.waitKey(1) < 0:
        # 从相机获取帧数据
        if orbbec_cap.grab():
            # 解码grab()获取的帧数据
            # rgb数据
            ret_bgr, bgr_image = orbbec_cap.retrieve(flag=cv.CAP_OBSENSOR_BGR_IMAGE)

            # 深度数据
            ret_depth, depth_map = orbbec_cap.retrieve(flag=cv.CAP_OBSENSOR_DEPTH_MAP)
            if ret_depth:
                color_depth_map = cv.normalize(depth_map, None, 0, 255, cv.NORM_MINMAX, cv.CV_8UC1)
                color_depth_map = cv.applyColorMap(color_depth_map, cv.COLORMAP_JET)
                cv.imshow("Depth",  color_depth_map)
                
                # 根据设置的阈值将深度图二值化，人体距相机的距离在thres的上下限之间
                segment_image, body_contour = segment_body(depth_map, thres)
                if ret_bgr:
                    cv.imshow("Segmentation", segment_image)
                    if body_contour is not None:
                        cv.drawContours(bgr_image, body_contour, -1, (0, 255, 0), 2, cv.LINE_AA)
                        cv.imshow("Body Contour", bgr_image)
        else:
            print("Fail to get data from camera!")
            break

def segment_body(depth_image, thres):
    # 人体与相机的距离在thres[0]和thres[1]之间
    output = cv.inRange(depth_image, thres[0], thres[1])
    output = cv.dilate(output, None)
    output = cv.dilate(output, None)

    # 寻找连通区域
    contours, hierarchy = cv.findContours(output, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    max_area = 0
    max_idx = -1
    # 取面积最大的为人体
    for i in range(len(contours)):
        area = cv.contourArea(contours[i])
        if area > max_area:
            max_area = area
            max_idx = i
    
    body_contour = None
    if max_idx > -1:
        body_contour = (contours[max_idx],)

    return output, body_contour


if __name__ == '__main__':
    # 假定人体与相机的距离范围在700mm～900mm之间
    thres = [700, 900]
    main(thres)
