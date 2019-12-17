#! /usr/bin/env python3


# -*- coding: utf8 -*-
import copy
import imutils
import cv2
import numpy as np
from keras.models import load_model
import time
import random
from utils.load_tf import I3D_Model
import os
# Cac khai bao bien
prediction = ''
score = 0


gesture_names = {0: 'l',
                 1: 'e',
                 2: 'f',
                 3: 'v',
                 4: 'b',
                 5: 'u',
                 6: 'c',
                 7: 'i',
                 8: 'o',
                 9: 'w',
                 10: 'x',
                 11: 'a',
                 12: 'g',
                 13: 'h',
                 14: 'y',
                 15: 'dd',
                 16: 'k',
                 17: 'm',
                 18: 'n',
                 19: 'p',
                 20: 'q',
                 21: 'r',
                 22: 's',
                 23: 't',
                 24: 'non',
                 25: 'moc',
                 26:'space',
                 27:'j',
                 28:'d'               
                 
                 }

# Ham de predict xem la ky tu gi
def predict_rgb_image_vgg(model, image):
    image = np.array(image, dtype='float32')
    image /= 255
    pred_array = model.predict(image)
    #print(f'pred_array: {pred_array}')
    result = gesture_names[np.argmax(pred_array)]
  
    score = float("%0.2f" % (max(pred_array[0]) * 100))
   
    return result, score


# Ham xoa nen khoi anh
def remove_background(bgModel , frame):
    fgmask = bgModel.apply(frame, learningRate=learningRate)
    kernel = np.ones((3, 3), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    res = cv2.bitwise_and(frame, frame, mask=fgmask)
    return res


# Khai bao kich thuoc vung detection region
cap_region_x_begin = 0.5
cap_region_y_end = 0.8

# Cac thong so lay threshold
threshold = 60
blurValue = 41
bgSubThreshold = 50#50
learningRate = 0

# Nguong du doan ky tu
predThreshold= 85

isBgCaptured = 0  # Bien luu tru da capture background chua
issavedata=0
# def processBacground():
    
#     global bgModel
#     bgModel = cv2.createBackgroundSubtractorMOG2()

# def reset():
#     global bgModel
#     bgModel = None

def predict(model, background, object_, bgModel):

    fgMask1 = remove_background(bgModel, background)
    
    fgMask2 = remove_background(bgModel, object_)
    fgMask2 = cv2.cvtColor(fgMask2, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(fgMask2, (blurValue, blurValue), 0)

    ret, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    thresh = np.flip(thresh, axis = 1)
    save = thresh

    # os.remove("thresh.jpg")
    # cv2.imwrite('thresh.jpg', thresh)
    # cv2.imshow('thresh', thresh)
    target = np.stack((thresh,) * 3, axis=-1)
    target = cv2.resize(target, (224, 224))
    target = target.reshape(1, 224, 224, 3)
    prediction, score = predict_rgb_image_vgg(model, target)
    thresh = None
    #print(score,prediction)
    if score>=predThreshold:
        return (score, prediction, save)
    return (None, None, save)

def findMax(detect1):
    scoreMax1 = 0
    predictO1 = None
    for (score, prediction) in detect1:
        if score is not None and score > scoreMax1:
            scoreMax1 = score
            predictO1 = prediction
    return (scoreMax1,predictO1)

class Detector:
    
    def __init__ (self):

        self.model = I3D_Model('models')

    def processToPredict(self, data):

        # chọn 1 frame từ 0 - 30 là background 
        background = data[10]

        # hành động có thể đưa ra dự đoán là kí tự có dấu nên chia thành 2 phần frame là từ 20 - 120

        object1 = random.randint(60,70) 
        
        object2 = random.randint(100,110) 

        bgModel = cv2.createBackgroundSubtractorMOG2(bgSubThreshold)
        res1 = predict(self.model, background, data[object1], bgModel)

        del bgModel
        bgModel = cv2.createBackgroundSubtractorMOG2(bgSubThreshold)
        res2 = predict(self.model, background, data[object2], bgModel)
        #print(res1, res2)
        # tìm kết quả dự đoán nhiều nhất trong từng nhóm detetc và đưa và kết quả cuối của nhóm đó.
        if res1[1] is None:
            return res2[1]
        if res2[1] is None:
            return res1[1]
        if res1[1] == res2[1]:
            return res1[1]
        else:
            if res1[1] == 'A':
                if res2[1] == 'non':
                    return 'Â'
                if res2[1] == 'moc':
                    return 'Ă'
                return 'A'
            if res1[1] == 'O':
                if res2[1] == 'non':
                    return 'Ô'
                if res2[1] == 'moc':
                    return 'Ơ'
                return 'O'
            if res1[1] == 'U':
                if res2[1] == 'moc':
                    return 'Ư'
                return 'U'
            if res1[1] == 'E':
                if res2[1] == 'non':
                    return 'Ê'
                return 'E'
        return res1[1]

    def predict_a_image(self, background, action ):

        # print("Tính toán ")
        bgModel = cv2.createBackgroundSubtractorMOG2(bgSubThreshold)
        res = predict(self.model, background, action, bgModel)
        # print(res)
        return res
# import os
# import cv2
# data = []
# model = load_model('models/mymodel.h5')
# path = 'data'
# for filename in os.listdir(path):
#     path_file = os.path.join(path, filename)
#     img  = cv2.imread(path_file)
#     data.append(img)

# print(processToPredict(data, model))

# model = load_model('models/mymodel.h5')
# a = [46, 48, 80, 79, 79]
# b = [130, 123, 129, 106, 115]
# c = [118, 112, 129, 127, 121]
# bgModel = cv2.createBackgroundSubtractorMOG2()
# print(predict(model, data[9], data[125], bgModel))
# background = cv2.imread('data/frame10.jpg')
# object_ = cv2.imread('data/frame126.jpg')
# bgModel2 = cv2.createBackgroundSubtractorMOG2()
# print(predict(model, background, object_, bgModel))
