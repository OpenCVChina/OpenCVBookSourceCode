#!/usr/bin/env python3
# encoding:utf-8


import argparse

import numpy as np
import cv2 as cv

def str2bool(v):
    if v.lower() in ['on', 'yes', 'true', 'y', 't']:
        return True
    elif v.lower() in ['off', 'no', 'false', 'n', 'f']:
        return False
    else:
        raise NotImplementedError

parser = argparse.ArgumentParser()
parser.add_argument('--image', '-i', type=str, help='Path to the input image1. Omit for detecting on default camera.')
parser.add_argument('--video', '-v', type=str, help='Path to the input video.')
parser.add_argument('--scale', '-sc', type=float, default=1.0, help='Scale factor used to resize input video frames.')
parser.add_argument('--face_detection_model', '-fd', type=str, default='face_detection_yunet_2023mar.onnx', help='Path to the face detection model. Download the model at https://github.com/opencv/opencv_zoo/tree/master/models/face_detection_yunet')
parser.add_argument('--score_threshold', type=float, default=0.9, help='Filtering out faces of score < score_threshold.')
parser.add_argument('--nms_threshold', type=float, default=0.3, help='Suppress bounding boxes of iou >= nms_threshold.')
parser.add_argument('--top_k', type=int, default=5000, help='Keep top_k bounding boxes before NMS.')
args = parser.parse_args()

def visualize(input, faces, fps, thickness=2):
    if faces[1] is not None:
        for idx, face in enumerate(faces[1]):
            print('Face {}, top-left coordinates: ({:.0f}, {:.0f}), box width: {:.0f}, box height {:.0f}, score: {:.2f}'.format(idx, face[0], face[1], face[2], face[3], face[-1]))

            coords = face[:-1].astype(np.int32)
            cv.rectangle(input, (coords[0], coords[1]), (coords[0]+coords[2], coords[1]+coords[3]), (0, 255, 0), thickness)
            cv.circle(input, (coords[4], coords[5]), 2, (255, 0, 0), thickness)
            cv.circle(input, (coords[6], coords[7]), 2, (0, 0, 255), thickness)
            cv.circle(input, (coords[8], coords[9]), 2, (0, 255, 0), thickness)
            cv.circle(input, (coords[10], coords[11]), 2, (255, 0, 255), thickness)
            cv.circle(input, (coords[12], coords[13]), 2, (0, 255, 255), thickness)
    cv.putText(input, 'FPS: {:.2f}'.format(fps), (1, 16), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

if __name__ == '__main__':

    ## [初始化FaceDetectorYN]
    detector = cv.FaceDetectorYN.create(
        args.face_detection_model,
        "",
        (320, 320),
        args.score_threshold,
        args.nms_threshold,
        args.top_k
    )
    ## [初始化FaceDetectorYN]

    tm = cv.TickMeter()

    # 若输入为图像
    if args.image is not None:
        img1 = cv.imread(args.image)
        img1Width = int(img1.shape[1]*args.scale)
        img1Height = int(img1.shape[0]*args.scale)

        img1 = cv.resize(img1, (img1Width, img1Height))
        tm.start()

        ## [推理]
        # 推理前需要设置输入大小
        detector.setInputSize((img1Width, img1Height))

        faces1 = detector.detect(img1)
        ## [推理]

        tm.stop()
        assert faces1[1] is not None, 'Cannot find a face in {}'.format(args.image1)

        # 将结果绘制在图像上
        visualize(img1, faces1, tm.getFPS())

        # 显示结果
        cv.imshow("yunet face", img1)
        cv.waitKey()        
    else: # 若输入为摄像头
        if args.video is not None:
            deviceId = args.video
        else:
            deviceId = 0
            # 打开深度相机。如果失败，修改参数为0，1，2中的某个值，继续尝试
            orbbec_cap = cv.VideoCapture(deviceId, cv.CAP_OBSENSOR)
            if orbbec_cap.isOpened() == False:
                print("Fail to open obsensor capture.")
                exit(0)

            #frameWidth = int(orbbec_cap.get(cv.CAP_PROP_FRAME_WIDTH)*args.scale)
            #frameHeight = int(orbbec_cap.get(cv.CAP_PROP_FRAME_HEIGHT)*args.scale)
            # Gemini2 colour profile
            frameWidth = int(640 * args.scale)
            frameHeight = int(480 * args.scale)
            detector.setInputSize([frameWidth, frameHeight])

            while cv.waitKey(1) < 0:
                # 从相机获取帧数据
                if orbbec_cap.grab():
                    # print("Grabbing data succeeds.")

                    # 解码grab()获取的帧数据
                    # rgb数据
                    ret_bgr, frame = orbbec_cap.retrieve(None, cv.CAP_OBSENSOR_BGR_IMAGE)
                    if ret_bgr:
                        frame = cv.resize(frame, (frameWidth, frameHeight))

                        # 推理
                        tm.start()
                        faces = detector.detect(frame) # faces是tuple类型
                        tm.stop()

                        # 将结果绘制在图像上
                        visualize(frame, faces, tm.getFPS())

                        # 显示结果
                        cv.imshow('Live', frame)
        orbbec_cap.release()
    cv.destroyAllWindows()
