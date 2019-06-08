
from PIL import Image
import os, glob
import numpy as np
from sklearn.model_selection import train_test_split

caltech_dir = "./image/101"
categories = ["broken", "dirty", "panel"]
nb_classes = len(categories)

#이미지 크기 지정5 
image_w = 64
image_h = 64
pixels = image_w*image_h*3  #3은 RGB 픽셀 개수 저장

#이미지 데이터 읽어 들이기
x_train = []
y_train=[]
x_test=[]
y_test=[]

for idx, cat in enumerate(categories):
    #레이블 지정
    label = [0 for i in range(nb_classes)]
    label[idx] = 1

    #이미지
    image_dir = caltech_dir+"/"+cat
    files=glob.glob(image_dir+"/*") #폴더 내부 모든 파일 읽어보기
    for i,f in enumerate(files):
        length = len(files)
        img = Image.open(f)
        img = img.convert("RGB")
        img = img.resize((image_w,image_h))
        data = np.asarray(img)
        if i < length * 0.7 :
            x_train.append(data)
            y_train.append(label)
        else:
            x_test.append(data)
            y_test.append(label)
        for angle in range(-20, 20, 5):
            #회전 데이터 추가
            img2 = img.rotate(angle)
            data = np.asarray(img2)
            if i < length * 0.7:
                x_train.append(data)
                y_train.append(label)

            #반전 데이터
            img2 = img2.transpose(Image.FLIP_LEFT_RIGHT)
            data = np.asarray(img2)
            if i < length * 0.7:
                x_train.append(data)
                y_train.append(label)



x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)
print("x_train", len(x_train))
print("y_train ", len(y_train ))
print("x_test", len(x_test))
print("y_test", len(y_test))

# x_train, x_test, y_train, y_test = \
#     train_test_split(X, Y)
xy = (x_train,x_test, y_train, y_test)
np.save("./image/solar.npy", xy)
#print("ok", len(Y))






import numpy as np
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import Convolution2D, MaxPooling2D
from keras.models import Sequential
from PIL import Image


# 카테고리 지정하기
categories = ["broken", "dirty", "panel"]
nb_classes = len(categories)

# 이미지 크기 지정하기
image_w = 64
image_h = 64

# 데이터 열기
x_train, x_test, y_train, y_test = np.load("./image/solar.npy")

# 데이터 정규화하기 (입력데이터는 0에서 1값을 가져야한다.)
x_train = x_train.astype("float") / 256
x_test = x_test.astype("float") / 256
print('x_train shape', x_train.shape)

# 모델 구축하기
model = Sequential()
model.add(Convolution2D(32, 3, 3,
                        border_mode='same',
                        input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Convolution2D(64, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

#모델 훈련하기
model.fit(x_train, y_train, batch_size=32, nb_epoch=100)

#모델 평가하기
score = model.evaluate(x_test, y_test)
print('loss=', score[0])
print('accuracy', score[1])
print(score)

#모델 저장하기
model.save('solar_model.h5')






#모델 예측하기
from keras.models import load_model
from keras.preprocessing import image
import numpy as np


img_width, img_height = 64, 64

model=load_model('solar_model.h5')
model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])


img1 = image.load_img('./image/cut_image/image.png',target_size=(img_width, img_height))
x1= image.img_to_array(img1)
x1=np.expand_dims(x1,axis=0)

images = np.vstack([x1])
classes = model.predict_classes(images, batch_size=32)

print(classes)
if classes == 0:
    print("PANEL1: There is a broken panel.")
if classes == 1:
    print("PANEL1: There is a dirty panel.")
if classes == 2:
    print("PANEL1: There is nothing on panel.")


