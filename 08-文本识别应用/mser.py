#!/usr/bin/env python3
# encoding:utf-8


import cv2 as cv
import numpy as np

 
def nms(boxes, overlapThresh):
    if len(boxes) == 0:
        return []
 
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
 
    pick = []
 
    # 取四个坐标数组
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
 
    # 计算面积数组
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
 
    # 按得分排序（如没有置信度得分，可按坐标从小到大排序，如右下角坐标）
    idxs = np.argsort(y2)
 
    # 开始遍历，并删除重复的框
    while len(idxs) > 0:
        # 将最右下方的框放入pick数组
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
 
        # 找剩下的其余框中最大坐标和最小坐标
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
 
        # 计算重叠面积占对应框的比例，即 IoU
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        overlap = (w * h) / area[idxs[:last]]
 
        # 如果 IoU 大于指定阈值，则删除
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))
 
    return boxes[pick].astype("int")

def mser(image):
 
    img_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
 
    mser = cv.MSER_create()
    regions, _ = mser.detectRegions(img_gray)

    hulls = [cv.convexHull(p.reshape(-1, 1, 2)) for p in regions]
    # 不规则轮廓
    #cv.polylines(image, hulls, 1, (255, 255, 0))
 
    keep = []
    for hull in hulls:
        x, y, w, h = cv.boundingRect(hull)
        keep.append([x, y, x + w, y + h])
        # 矩形框
        #cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 1)

    boxes = nms(np.array(keep), 0.4)
    for box in boxes:
        # NMS后的矩形框
        cv.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 1)
    

def main():
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print('Failed to open camera.')
        exit(0)

    while cv.waitKey(1) < 0:
        hasFrame, frame = cap.read()
        if not hasFrame:
            print('Failed to read frame.')            
            break

        mser(frame) 
        cv.imshow("bounding box", frame)

    cap.release()
    cv.destroyAllWindows()   
 
 
if __name__ == '__main__':
    main()
