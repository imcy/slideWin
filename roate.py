# -*- coding:utf-8 -*-
import cv2
from math import *
import numpy as np
import time,math
import os
import re

'''旋转图像并剪裁'''
def rotate(
        img,  # 图片
        pt1, pt2, pt3, pt4
):
    print(pt1,pt2,pt3,pt4)
    withRect = math.sqrt((pt4[0] - pt1[0]) ** 2 + (pt4[1] - pt1[1]) ** 2)  # 矩形框的宽度
    heightRect = math.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) **2)
    print(withRect,heightRect)
    angle = acos((pt4[0] - pt1[0]) / withRect) * (180 / math.pi)  # 矩形框旋转角度
    print(angle)

    if pt4[1]>pt1[1]:
        print("顺时针旋转")
    else:
        print("逆时针旋转")
        angle=-angle

    height = img.shape[0]  # 原始图像高度
    width = img.shape[1]   # 原始图像宽度
    rotateMat = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)  # 按angle角度旋转图像
    heightNew = int(width * fabs(sin(radians(angle))) + height * fabs(cos(radians(angle))))
    widthNew = int(height * fabs(sin(radians(angle))) + width * fabs(cos(radians(angle))))

    rotateMat[0, 2] += (widthNew - width) / 2
    rotateMat[1, 2] += (heightNew - height) / 2
    imgRotation = cv2.warpAffine(img, rotateMat, (widthNew, heightNew), borderValue=(255, 255, 255))
    cv2.imshow('rotateImg2',  imgRotation)
    cv2.waitKey(0)

    # 旋转后图像的四点坐标
    [[pt1[0]], [pt1[1]]] = np.dot(rotateMat, np.array([[pt1[0]], [pt1[1]], [1]]))
    [[pt3[0]], [pt3[1]]] = np.dot(rotateMat, np.array([[pt3[0]], [pt3[1]], [1]]))
    [[pt2[0]], [pt2[1]]] = np.dot(rotateMat, np.array([[pt2[0]], [pt2[1]], [1]]))
    [[pt4[0]], [pt4[1]]] = np.dot(rotateMat, np.array([[pt4[0]], [pt4[1]], [1]]))

    # 处理反转的情况
    if pt2[1]>pt4[1]:
        pt2[1],pt4[1]=pt4[1],pt2[1]
    if pt1[0]>pt3[0]:
        pt1[0],pt3[0]=pt3[0],pt1[0]

    imgOut = imgRotation[int(pt2[1]):int(pt4[1]), int(pt1[0]):int(pt3[0])]
    #cv2.imshow("imgOut", imgOut)  # 裁减得到的旋转矩形框
    #cv2.waitKey(0)
    return imgOut  # rotated image



def drawRect(img,pt1,pt2,pt3,pt4,color,lineWidth):
    cv2.line(img, pt1, pt2, color, lineWidth)
    cv2.line(img, pt2, pt3, color, lineWidth)
    cv2.line(img, pt3, pt4, color, lineWidth)
    cv2.line(img, pt1, pt4, color, lineWidth)

def ReadTxt(directory,imageName,last,cls):
    fileTxt=directory+"//rawLabel//"+imageName[:7]+last  # txt文件名
    imagePath=directory+'//'+imageName
    getTxt=open(fileTxt, 'r')  # 打开txt文件
    lines = getTxt.readlines()
    length=len(lines)
    for i in range(0,length,4):
        pt2=list(map(float,lines[i].split(' ')[:2]))
        pt1=list(map(float,lines[i+1].split(' ')[:2]))
        pt4=list(map(float,lines[i+2].split(' ')[:2]))
        pt3=list(map(float,re.split('\n| ',lines[i+3])[:2]))
        # float转int
        pt2=list(map(int,pt2))
        pt1=list(map(int,pt1))
        pt4=list(map(int,pt4))
        pt3=list(map(int,pt3))

        imgSrc = cv2.imread(imagePath)
        drawRect(imgSrc, tuple(pt1),tuple(pt2),tuple(pt3),tuple(pt4), (0, 0, 255), 2)
        cv2.imshow("img", imgSrc)
        cv2.waitKey(0)
        # 保存剪裁图片
        imgCut=rotate(imgSrc,pt1,pt2,pt3,pt4)
        #savedir = './/negCut//test//'+str(cls)+'//'+imageName[:7]+'-'+str(i/4)+'.png'
        #print savedir
        #cv2.imwrite(savedir, imgCut)

if __name__=="__main__":
    last = 'cneg.txt'
    for i in range(11,15):
        directory = "G://grasp//grapCode//testImage//" + str(i) + "//"
        print(directory)
        for filename in os.listdir(directory):  # listdir的参数是文件夹的路径
            if filename.endswith('png'):
                print(filename)
                ReadTxt(directory, filename, last,i)

    #imageName="pcd0247r.png"
    #ReadTxt(directory,imageName,last)
    '''
    imgSrc=cv2.imread('./pcd0247r.png')
    imgResize=imgSrc
    pt1 = (281,244)
    pt2 = (286,219)
    pt3 = (313,224)
    pt4 = (308,249)

    drawRect(imgResize, pt1, pt2, pt3, pt4, (0, 0, 255), 2)
    cv2.imshow("img", imgResize)
    cv2.waitKey(0)
    imgRotation=rotate(imgResize,pt1, pt2, pt3, pt4)
    '''


