import cv2 as cv
import numpy as np


def main():
    # 读入图像，并转为灰度图
    # 目标物体图像
    img1 = cv.imread('box.png')
    img1_gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    # 要寻找目标物体的场景图像
    img2 = cv.imread('box_in_scene.png')
    img2_gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

    # 提取SIFT特征并计算描述子
    sift = cv.SIFT.create()
    kp1, des1 = sift.detectAndCompute(img1_gray, None)
    kp2, des2 = sift.detectAndCompute(img2_gray, None)

    # FLANN参数
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv.FlannBasedMatcher(index_params, search_params)

    # FLANN特征匹配
    matches = flann.knnMatch(des1, des2, k = 2)

    # 根据ratio保留好的匹配对
    good_matches = []
    for m,n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    # 如果好的特征匹配对数量足够多，则计算单应性矩阵
    # 这里设置为大于10
    if len(good_matches) > 10:
        # 整理两幅图像中匹配特征对应的点
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)

        # 计算目标物体图像到场景图像中目标物体的单应性矩阵
        H, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
        # 正常特征的点（inliers）
        matchesMask = mask.ravel().tolist()
        
        # 目标物体的坐标
        h, w, d = img1.shape
        obj_corners = np.float32([[0,0], [0,h-1], [w-1,h-1], [w-1,0]]).reshape(-1,1,2)

    	# 根据单应性矩阵计算目标物体在场景图像中的坐标    
        scene_corners = cv.perspectiveTransform(obj_corners, H)

    	# 将在场景图像中找到的目标物体用红色的框标记出来
        img2 = cv.polylines(img2, [np.int32(scene_corners)], True, (0,0,255), 2, cv.LINE_AA)
    else:
        print( "匹配的特征对数量不够多 - 找到的匹配对：{}/最少需要的匹配对：{}".format(len(good), MIN_MATCH_COUNT) )
        matchesMask = None

    # 绘制匹配的特征对
    draw_params = dict(matchColor = (0,255,0), 
                       matchesMask = matchesMask, 
                       flags = cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    img3 = cv.drawMatches(img1, kp1, img2, kp2, good_matches, None, **draw_params)

    cv.imwrite('find.jpg', img3)
    cv.imshow('application', img3)
    cv.waitKey()


if __name__ == '__main__':
    main()
