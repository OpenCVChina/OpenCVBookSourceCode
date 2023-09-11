#!/usr/bin/env python3
# encoding:utf-8


import os
import argparse

import numpy as np
import cv2 as cv

parser = argparse.ArgumentParser()
parser.add_argument('--database_dir', '-db', type=str, default='./database')
parser.add_argument('--face_detection_model', '-fd', type=str, default='face_detection_yunet_2023mar.onnx')
parser.add_argument('--face_recognition_model', '-fr', type=str, default='face_recognition_sface_2021dec.onnx')
args = parser.parse_args()


def detect_face(detector, image):
    ''' 对image进行人脸检测
    '''
    h, w, c = image.shape
    if detector.getInputSize() != (w, h):
        detector.setInputSize((w, h))

    faces = detector.detect(image)
    return [] if faces[1] is None else faces[1]

def extract_feature(recognizer, image, faces):
    ''' 根据faces中的人脸框进行人脸对齐; 从对齐后的人脸提取特征
    '''
    features = []

    for face in faces:
        aligned_face = recognizer.alignCrop(image, face)
        feature = recognizer.feature(aligned_face)
        features.append(feature)
    return features

def match(recognizer, feature1, feature2, dis_type=1):
    l2_threshold = 1.128
    cosine_threshold = 0.363

    score = recognizer.match(feature1, feature2, dis_type)
    # print(score)
    if dis_type == 0: # Cosine相似度
        if score >= cosine_threshold:
            return True
        else:
            return False
    elif dis_type == 1: # L2距离
        if score > l2_threshold:
            return False
        else:
            return True
    else:
        raise NotImplementedError('dis_type = {} is not implemenented!'.format(dis_type))

def load_database(database_path, detector, recognizer):
    db_features = dict()

    print('Loading database ...')
    # 首先读入已提取的人脸特征
    for filename in os.listdir(database_path):
        if filename.endswith('.npy'):
            identity = filename[:-4]
            if identity not in db_features:
                db_features[identity] = np.load(os.path.join(database_path, filename))
    npy_cnt = len(db_features)
    # 读入图像并提取人脸特征
    for filename in os.listdir(database_path):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            identity = filename[:-4]
            if identity not in db_features:
                image = cv.imread(os.path.join(database_path, filename))
                faces = detect_face(detector, image)
                features = extract_feature(recognizer, image, faces)
                if len(features) > 0:
                    db_features[identity] = features[0]
                    np.save(os.path.join(database_path, '{}.npy'.format(identity)), features[0])
    cnt = len(db_features)
    print('Database: {} loaded in total, {} loaded from .npy, {} loaded from images.'.format(cnt, npy_cnt, cnt-npy_cnt))
    return db_features

def visualize(image, faces, identities, fps, box_color=(0, 255, 0), text_color=(0, 0, 255)):
    output = image.copy()

    # 在左上角绘制帧率
    cv.putText(output, 'FPS: {:.2f}'.format(fps), (0, 15), cv.FONT_HERSHEY_DUPLEX, 0.5, text_color)

    for face, identity in zip(faces, identities):
        # 绘制人脸框
        bbox = face[0:4].astype(np.int32)
        cv.rectangle(output, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), box_color, 2)
        # 绘制识别结果
        cv.putText(output, '{}'.format(identity), (bbox[0], bbox[1]-15), cv.FONT_HERSHEY_DUPLEX, 0.5, text_color)

    return output

if __name__ == '__main__':
    # 初始化FaceDetectorYN
    detector = cv.FaceDetectorYN.create(
        args.face_detection_model,
        "",
        (640, 480),
        score_threshold=0.9,
    )
    # 初始化FaceRecognizerSF
    recognizer = cv.FaceRecognizerSF.create(
        args.face_recognition_model,
        ""
    )

    # 读入数据库
    database = load_database(args.database_dir, detector, recognizer)

    # 初始化视频流
    device_id = 0
    # 打开深度相机。如果失败，修改参数为0，1，2中的某个值，继续尝试
    orbbec_cap = cv.VideoCapture(device_id, cv.CAP_OBSENSOR)
    if orbbec_cap.isOpened() == False:
        print("Fail to open obsensor capture.")
        exit(0)
    w = int(orbbec_cap.get(cv.CAP_PROP_FRAME_WIDTH))
    h = int(orbbec_cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    # 实时人脸识别
    tm = cv.TickMeter()

    while cv.waitKey(1) < 0:
        # frame = cv.imread('gemma4.jpg')
        # 从相机获取帧数据
        if orbbec_cap.grab():
            # print("Grabbing data succeeds.")

            # 解码grab()获取的帧数据
            # rgb数据
            ret_bgr, frame = orbbec_cap.retrieve(None, cv.CAP_OBSENSOR_BGR_IMAGE)
            if ret_bgr:
                tm.start()
                # 人脸检测
                faces = detect_face(detector, frame)
                # 特征提取额
                features = extract_feature(recognizer, frame, faces)
                # 与数据库进行人脸比对
                identities = []
                for feature in features:
                    isMatched = False
                    for identity, db_feature in database.items():
                        isMatched = match(recognizer, feature, db_feature)
                        if isMatched:
                            identities.append(identity)
                            break
                    if not isMatched:
                        identities.append('Unknown')
                tm.stop()

                # 将结果绘制在图像上
                frame = visualize(frame, faces, identities, tm.getFPS())

                # 显示结果
                cv.imshow('Face recognition system', frame)

                tm.reset()

    orbbec_cap.release()
    cv.destroyAllWindows() 
