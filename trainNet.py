# -*- coding: utf-8 -*-
from keras.models import Sequential
from keras.layers.core import Dense,Dropout,Activation
from keras.optimizers import SGD
import os
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.externals import joblib
import glob
import os
import random


train_feat_path = './features/train'
train_fds = []  # 训练特征
train_labels = []  # 训练标签
train_data=[] # 训练数据的特征+标签
train_num=0
for feat_path in glob.glob(os.path.join(train_feat_path, '*.feat')):
    train_num+=1
    data = joblib.load(feat_path)
    train_data.append(data)
    print("%d Dealing with %s train" % (train_num, feat_path))
    random.shuffle(train_data)
for data in train_data:
    train_fds.append(data[:-1])
    train_labels.append(data[-1])


test_feat_path = './features/test'
test_fds = []  # 测试特征
test_labels = []  # 测试标签
test_data= []  # 测试数据特征+标签
test_num= 0
for feat_path in glob.glob(os.path.join(test_feat_path, '*.feat')):
    test_num +=1
    data = joblib.load(feat_path)
    test_data.append(data)
    print("%d Dealing with %s train" % (test_num, feat_path))
    random.shuffle(test_data)
for data in test_data:
    test_fds.append(data[:-1])
    test_labels.append(data[-1])


model=Sequential()#model initial

model.add(Dense(300, input_dim=576, init='uniform'))#2330 input，Hidden layer has 300 unit
model.add(Activation('tanh')) #Hidden layer activate function is tanh
model.add(Dropout(0.5))
model.add(Dense(1, init='uniform')) #1 output
model.add(Activation('sigmoid'))#output layer activate function is sigmoid

sgd=SGD(lr=0.01,decay=1e-6,momentum=0.9,nesterov=True) #Using gradient descent algorithm
model.compile(loss='mean_squared_error',optimizer=sgd,metrics=["accuracy"]) #Compile model
model.fit(np.array(train_fds),np.array(train_labels),nb_epoch=75,batch_size=50)
loss, accuracy=model.evaluate(np.array(test_fds),np.array(test_labels),verbose=1)
print("\n")
print("Accuracy = {:.2f}".format(accuracy))
print("loss=",loss)

predit_y=model.predict(np.array(test_fds))
fpr, tpr, thresholds = roc_curve(test_labels, predit_y, pos_label=1) #calculate the roc curve
from sklearn.metrics import auc
print("Auc=",auc(fpr, tpr)) #calculate the auc
plt.plot(fpr,tpr) #draw the roc curve

model_path = './models/net.model'
model.save(model_path)
plt.show()