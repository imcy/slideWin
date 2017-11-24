# -*- coding:utf-8 -*-
from skimage.feature import hog
from sklearn.externals import joblib
import numpy as np
from PIL import Image
import cv2
import os

# define parameter
normalize = True
visualize = False
block_norm = 'L2-Hys'
cells_per_block = [2, 2]
pixels_per_cell = [20, 20]
orientations = 9

def getDataWithCrop(filePath, label):
    Data = []
    num = 0
    for childDir in os.listdir(filePath):
        f_im = os.path.join(filePath, childDir)
        region = Image.open(f_im)  # open the image
        data = np.asarray(region)  # put the data of image into an N-dinimeter array
        data = cv2.resize(data, (100, 100), interpolation=cv2.INTER_CUBIC)  # resize image
        data = np.reshape(data, (100 * 100, 3))
        data.shape = 1, 3, -1
        fileName = np.array([[childDir]])
        datalebels = zip(data, label, fileName)  # organise data
        Data.extend(datalebels)  # pou the organised data into a list
        num += 1
        print("%d processing: %s" % (num, childDir))
    return Data, num


def getData(filePath, label):  # get the full image without cutting
    Data = []
    num = 0
    for childDir in os.listdir(filePath):
        f = os.path.join(filePath, childDir)
        data = cv2.imread(f)
        data = cv2.resize(data, (200, 200), interpolation=cv2.INTER_CUBIC)
        data = np.reshape(data, (200 * 200, 3))
        data.shape = 1, 3, -1
        fileName = np.array([[childDir]])
        datalebels = zip(data, label, fileName)
        Data.extend(datalebels)
        num += 1
        print("%d processing: %s" % (num, childDir))
    return Data, num


def getFeat(Data, mode):  # get and save feature valuve
    num = 0
    for data in Data:
        image = np.reshape(data[0], (100, 100, 3))
        gray = rgb2gray(image) / 255.0  # trans image to gray
        fd = hog(gray, orientations, pixels_per_cell, cells_per_block, block_norm, visualize, normalize)
        fd = np.concatenate((fd, data[1]))  # add label in the end of the array
        filename = list(data[2])
        fd_name = filename[0].split('.')[0] + '.feat'  # set file name
        if mode == 'train':
            fd_path = os.path.join('./features/train/','neg-'+fd_name)
        else:
            fd_path = os.path.join('./features/test/', 'neg-'+fd_name)
        joblib.dump(fd, fd_path, compress=3)  # save data to local
        num += 1
        print("%d saving: %s." % (num, fd_name))


def rgb2gray(im):
    gray = im[:, :, 0] * 0.2989 + im[:, :, 1] * 0.5870 + im[:, :, 2] * 0.1140
    return gray


if __name__ == '__main__':
    # deal with Positive test dataset and trainset with cutting
    for i in range(15):
        Ptrain_filePath = './negCut/train/'+str(i)
        PTrainData, P_train_num = getDataWithCrop(Ptrain_filePath, np.array([[0]]))
        getFeat(PTrainData, 'train')