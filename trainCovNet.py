import numpy as np

np.random.seed(1337)  # for reproducibility
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
import pickle


# 全局变量  
batch_size = 128
nb_classes = 2
epochs = 12
# 输入图片尺寸
img_rows, img_cols = 100, 100
# number of convolutional filters to use  
nb_filters = 32
# size of pooling area for max pooling  
pool_size = (2, 2)
# convolution kernel size  
kernel_size = (3, 3)

file_train = open("./train/dataset_train.pkl",'rb')
X_train, Y_train = pickle.load(file_train)
file_test = open("./test/dataset_test.pkl",'rb')
X_test, Y_test = pickle.load(file_test)


# 根据不同的backend定下不同的格式  
if K.image_dim_ordering() == 'th':
    X_train = X_train.reshape(X_train.shape[0], 3, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 3, img_rows, img_cols)
    input_shape = (3, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 3)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 3)
    input_shape = (img_rows, img_cols, 3)
print(input_shape)


X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

'''
# 转换为one_hot类型  
Y_train = np_utils.to_categorical(Y_train, nb_classes)
print(Y_train.shape[0])
Y_test = np_utils.to_categorical(Y_test, nb_classes)
'''

# 构建模型  
model = Sequential()

model.add(Convolution2D(nb_filters, (kernel_size[0], kernel_size[1]),
                        padding='same',
                        input_shape=input_shape))  # 卷积层1  
model.add(Activation('relu'))  # 激活层  
model.add(Convolution2D(nb_filters, (kernel_size[0], kernel_size[1])))  # 卷积层2  
model.add(Activation('relu'))  # 激活层  
model.add(MaxPooling2D(pool_size=pool_size))  # 池化层  
model.add(Dropout(0.25))  # 神经元随机失活  
model.add(Flatten())  # 拉成一维数据  
model.add(Dense(128))  # 全连接层1  
model.add(Activation('relu'))  # 激活层  
model.add(Dropout(0.5))  # 随机失活  
model.add(Dense(nb_classes))  # 全连接层2  
model.add(Activation('softmax'))  # Softmax评分  

# 编译模型  
model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])
# 训练模型  
model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs,
          verbose=1, validation_data=(X_test, Y_test))
# 评估模型  
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])