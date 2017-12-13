# -*- coding: utf-8 -*-
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
import math

# 全局参数变量
normalize = True
visualize = False
block_norm = 'L2-Hys'
cells_per_block = [2,2]
pixels_per_cell = [20,20]
orientations = 9
batch_size = 128
nb_classes = 2
epochs = 12
# 输入图片尺寸
img_rows, img_cols = 100, 100
# 卷积核个数
nb_filters = 32
# 池化层大小
pool_size = (2, 2)
# 卷积核大小
kernel_size = (3, 3)
# 根据不同的backend定下不同的格式
input_shape = (img_rows, img_cols, 3)
scales = [(20, 20),(30,30),(40, 40), (100, 100)] # 滑动窗口尺寸

def sliding_window(image, stepSize, windowSize):
	# slide a window across the image
	for y in range(0, image.shape[0], stepSize):
		for x in range(0, image.shape[1], stepSize):
			# yield the current window
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

def rgb2gray(im):
    gray = im[:, :, 0]*0.2989+im[:, :, 1]*0.5870+im[:, :, 2]*0.1140
    return gray

def getModel():
    # 构建模型
    model = Sequential()
    model.add(Convolution2D(nb_filters, (kernel_size[0], kernel_size[1]),
                            padding='same',
                            input_shape=input_shape))  # 卷积层1
    model.add(Activation('relu'))  # 激活层
    #model.add(Convolution2D(nb_filters, (kernel_size[0], kernel_size[1])))  # 卷积层2
    #model.add(Activation('relu'))  # 激活层
    model.add(MaxPooling2D(pool_size=pool_size))  # 池化层
    model.add(Dropout(0.25))  # 神经元随机失活
    model.add(Flatten())  # 拉成一维数据
    model.add(Dense(128))  # 全连接层1
    model.add(Activation('relu'))  # 激活层
    model.add(Dropout(0.5))  # 随机失活
    model.add(Dense(nb_classes))  # 输出层
    model.add(Activation('softmax'))  # Softmax评分

    model_path = './models/convNets.h5'
    model.load_weights(model_path)
    return model

'''获取最佳抓取位置'''
def getBestGrasp():
    count=0
    max_win=0
    max_res=0
    maxSize=[]
    model = getModel()
    for (winW,winH) in scales:
        for (x, y, window) in sliding_window(image, stepSize=20, windowSize=(winW,winH)):
            if window.shape[0] != winH or window.shape[1] != winW:
                continue
            window = cv2.resize(window,(100,100),interpolation=cv2.INTER_CUBIC)
            window2=window/255.0
            window2 = np.array([np.array(window2)])
            res=model.predict_classes(window2)
            score=model.predict(window2)
            if res[0]==1:
                print(res, score[0][1])
                if score[0][1]>max_res:
                    max_res=score[0][1]
                    max_win = window
                    maxSize=[x,y,winW,winH]
                #cv2.imshow("asd", window)
                #cv2.waitKey(0)
            count+=1
    print(count)
    return max_win,max_res,maxSize


def getAngle(image,dsize):
    cv2.imshow("win", image)
    cv2.waitKey(0)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gradX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradY = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)
    # subtract the y-gradient from the x-gradient
    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)

    # blur and threshold the image
    blurred = cv2.blur(gradient, (9, 9))
    (_, thresh) = cv2.threshold(blurred, 90, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # perform a series of erosions and dilations
    closed = cv2.erode(closed, None, iterations=4)
    closed = cv2.dilate(closed, None, iterations=4)

    cv2.imshow("close", closed)
    cv2.waitKey(0)

    (_,cnts,_) = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]

    # 计算得到面积最大的轮廓
    rect = cv2.minAreaRect(c)
    box = np.int0(cv2.boxPoints(rect))

    # draw a bounding box arounded the detected barcode and display the image
    image = cv2.drawContours(image, [box], -1, (0, 255, 0), 3)

    box[0][0]=box[0][0]*dsize[0]
    box[1][0] = box[1][0] * dsize[0]
    box[2][0] = box[2][0] * dsize[0]
    box[3][0] = box[3][0] * dsize[0]

    box[0][1] = box[0][1] * dsize[1]
    box[1][1] = box[1][1] * dsize[1]
    box[2][1] = box[2][1] * dsize[1]
    box[3][1] = box[3][1] * dsize[1]

    width = math.sqrt((box[3][0] - box[0][0])** 2 + (box[3][1] - box[0][1]) ** 2)
    height = math.sqrt((box[1][0] - box[0][0]) ** 2 + (box[1][1] - box[0][1]) ** 2)
    angle = math.acos((box[3][0] - box[0][0]) / width) * (180 / math.pi)

    if height<width:
        angle+=90
        width=height

    cv2.imshow("angle", image)
    cv2.waitKey(0)

    print(angle,width,height)
    return angle, width,height

def draw(img,x,y,angle,width,height,diff):
    anglePi = -angle*math.pi/180.0
    cosA = math.cos(anglePi)
    sinA = math.sin(anglePi)

    x=x+diff[0]  # 与原始图片的偏差
    y=y+diff[1]
    height=23

    x1=x-0.5*width
    y1=y-0.5*height

    x0=x+0.5*width
    y0=y1

    x2=x1
    y2=y+0.5*height

    x3=x0
    y3=y2

    x0n= int((x0 -x)*cosA -(y0 - y)*sinA + x)
    y0n = int((x0-x)*sinA + (y0 - y)*cosA + y)

    x1n= int((x1 -x)*cosA -(y1 - y)*sinA + x)
    y1n = int((x1-x)*sinA + (y1 - y)*cosA + y)

    x2n= int((x2 -x)*cosA -(y2 - y)*sinA + x)
    y2n = int((x2-x)*sinA + (y2 - y)*cosA + y)

    x3n= int((x3 -x)*cosA -(y3 - y)*sinA + x)
    y3n = int((x3-x)*sinA + (y3 - y)*cosA + y)

    cv2.line(img,(x0n, y0n),(x1n, y1n),color=(0,0,255),thickness=2)
    cv2.line(img,(x1n, y1n), (x2n, y2n), color=(255,0,0),thickness=2)
    cv2.line(img,(x2n, y2n), (x3n, y3n), color=(0,0,255),thickness=2)
    cv2.line(img,(x0n, y0n), (x3n, y3n), color=(255,0,0),thickness=2)

    cv2.imshow("img",img)
    cv2.waitKey(0)

if __name__=="__main__":
    image_path = './cutImage/test/3/0pcd0392r.png'
    imageOrig_path = './cutImage/test/3/pcd0392r.png'
    image = cv2.imread(image_path) # 分割后图片
    imageOrig = cv2.imread(imageOrig_path)  # 原始图片

    orig = image.copy()  # 复制原始图片
    dsize = [image.shape[1]/100, image.shape[0]/100]  # 原图缩放后的缩放系数

    image = cv2.resize(image, (100, 100), interpolation=cv2.INTER_CUBIC)
    rects = []
    

    win,res,sizeRes=getBestGrasp() # 获取最佳抓取区域

    x=int(sizeRes[0]*dsize[0])  # 根据缩放系数获得原图像中对应x,y,winW,winH
    y=int(sizeRes[1]*dsize[1])
    winW = int(sizeRes[2]*dsize[0])
    winH = int(sizeRes[3]*dsize[1])

    getx=x+winW/2  # 获得中心点坐标x
    gety=y+winH/2  # 获得中心点坐标y

    win=orig[y:y + winH, x:x + winW]
    win=cv2.resize(win,(100,100),interpolation=cv2.INTER_CUBIC)
    dsize2 = [winW / 100, winH / 100]

    diff=[274,251] # 分割图片与原图片的偏差
    angle,width,height=getAngle(win,dsize2)  # 索贝尔算子获得角度
    draw(imageOrig,getx,gety,angle,winW,height,diff)  # 画出抓取区域




