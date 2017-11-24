# -*- coding: utf-8 -*-
import cv2
from skimage.feature import hog

# define parameter
normalize = True
visualize = False
block_norm = 'L2-Hys'
cells_per_block = [2,2]
pixels_per_cell = [20,20]
orientations = 9


def sliding_window(image, stepSize, windowSize):
	# slide a window across the image
	for y in xrange(0, image.shape[0], stepSize):
		for x in xrange(0, image.shape[1], stepSize):
			# yield the current window
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])


image_path = './pcd0135r.png'
image = cv2.imread(image_path)
cv2.imshow("img",image)
cv2.waitKey(0)

image = cv2.resize(image,(100,100),interpolation=cv2.INTER_CUBIC)
orig = image.copy()
orig = cv2.resize(orig,(100,100),interpolation=cv2.INTER_CUBIC)
rects = []
scales = [ (40,40),(50,50),(60,60),(image.shape[1],image.shape[0])]


def calculateIoU(candidateBound, groundTruthBound):
    cx1 = candidateBound[0]
    cy1 = candidateBound[1]
    cx2 = candidateBound[2]
    cy2 = candidateBound[3]

    gx1 = groundTruthBound[0]
    gy1 = groundTruthBound[1]
    gx2 = groundTruthBound[2]
    gy2 = groundTruthBound[3]

    carea = (cx2 - cx1) * (cy2 - cy1)  # C的面积
    garea = (gx2 - gx1) * (gy2 - gy1)  # G的面积

    x1 = max(cx1, gx1)
    y1 = max(cy1, gy1)
    x2 = min(cx2, gx2)
    y2 = min(cy2, gy2)
    w = max(0, x2 - x1)
    h = max(0, y2 - y1)
    area = w * h  # C∩G的面积

    iou = area / (carea + garea - area)

    return iou

def rgb2gray(im):
    gray = im[:, :, 0]*0.2989+im[:, :, 1]*0.5870+im[:, :, 2]*0.1140
    return gray

count=0
for (winW,winH) in scales:
    for (x, y, window) in sliding_window(image, stepSize=10, windowSize=(winW,winH)):
        result = 0
        if window.shape[0] != winH or window.shape[1] != winW:
            continue
        #cv2.imshow("asd", window)
        #cv2.waitKey(0)
        if window.shape[0] != 50 or window.shape[1] != 50:
            window = cv2.resize(window,(50,50),interpolation=cv2.INTER_CUBIC)
        gray = rgb2gray(window) / 255.0
        window_fd = hog(gray, orientations, pixels_per_cell, cells_per_block, block_norm, visualize, normalize)
        win_fd = window_fd.reshape(1, -1)
        print win_fd
        print win_fd.size
        cv2.rectangle(orig, (x, y), (x + winW, y + winH), (0, 0, 255), 2)
        count+=1
print count
#cv2.imshow("orig",orig)
#cv2.waitKey(0)

