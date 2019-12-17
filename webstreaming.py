from flask import Flask, render_template, Response, request, jsonify
import matplotlib.pyplot as plt
from imutils.video import VideoStream
from keras.models import load_model
from detection import Detector
from threading import Thread
import threading
import numpy as np
import cv2
import os
import io
from PIL import Image
import concurrent.futures

d = Detector()
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

data = []
count = 0
result = []

def capture():
    cap = cv2.VideoCapture(0)
    global count
    global data
    global d
    global result

    rest_txt = "None"
    # data = []
    bf       = None
    count_bf = 0
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Error: failed to capture image")
            break
        
        if count >= 20 and count %5 == 0:
            img  =  frame[62:320,2:260, :]
            count += 1
            res = d.predict_a_image(data[10], img)
            # print(res)
            img = res[2]
            cv2.imshow("img", img)
            if bf == None:
                bf = res[1]
                count_bf += 1
            if res[1] != bf:
                bf = res[1]
                count_bf = 1
            if bf == res[1]:
                count_bf += 1
            if bf == None:
                bf_count = 0
            if count_bf == 15 and bf != "None":
                cv2.putText(frame,"Done", (400,150), cv2.FONT_ITALIC, 1, (52, 219, 152), thickness = 2)
                result.append(bf)
                bf = "None"
                count_bf = 0
                 
                # print(result)
            rest_txt =  "Sign : " + str(res[1]) + " " + str(res[0]) + "%"
            cv2.putText(frame,rest_txt, (265,150), cv2.FONT_ITALIC, 1, (52, 219, 152), thickness = 2)
        
        elif count >= 20 and count %5 != 0:
            
            count += 1
            cv2.putText(frame,rest_txt, (265,150), cv2.FONT_ITALIC, 1, (52, 219, 152), thickness = 2)

        elif  count < 20:
            count += 1
            data.append(frame[62:320,2:260, :])
            cv2.putText(frame,"Capturing Background", (265,150), cv2.FONT_ITALIC, 1, (0, 0, 255), thickness = 2)

        elif count > 150:
            count = 20

        cv2.rectangle(frame, (0, 60), (262, 322), (255,0,0), 2)
        
        cv2.imwrite('demo.jpg', frame)
        
        k = cv2.waitKey(10)
        
        if k == ord('r'):
            data = []
            count = 0

        if k == ord('a'):
            if(len(result)>0):
                del(result[len(result)-1])
            
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

        yield (open('demo.jpg', 'rb').read())

def gen():
    
    generator = capture()
    for i in generator:
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + i + b'\r\n')


def capture2():
    while True:
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + open('demo2.jpg', 'rb').read() + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_thresh')
def video_thresh():
    return Response(capture2(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/export')
def export():
    global result
    if (len(result) <= 0):
        return render_template('index.html', result = "None")
    
    # print(result)
    s = ""
    bf = ""
    result.append(" ")
    result.append(" ")
    
    # for i in range(1, len(result) - 2):
    #     if result[i] == result[i-1]:
    #         del result[i]

    for i in range(len(result) - 1):
        character = str(result[i])

        if character == bf:
            continue
        if str(character) == "None":
            bf = "None"
            continue
        elif character == "space":
            s += " "
            bf = "space"
        elif character != bf and  character == 'dd':
            s += 'đ'
            bf = character

        elif character == 'a' and character != bf:
            tmp = str(result[i+1])
            bf = tmp
            if tmp == 'non':
                s += 'ă'
                i += 1
            elif tmp == 'moc':
                s += 'â'
                i += 1
            elif tmp == 'j':
                s += 'ạ'
                i += 1
                bf = 'ạ'
            elif tmp == 's':
                s += 'á'
                i+= 1
            elif tmp == 'f':
                s += 'à'
                i+= 1
            elif tmp == 'r':
                s += 'ả'
                i+=1 
            elif tmp == 'x':
                s += 'ã'
                i += 1
            else:
                s += 'a'
                bf = character
        
        elif character == 'o' and character != bf:
            tmp = str(result[i+1])
            bf = tmp
            if tmp == 'o':
                s += 'ô'
                i += 1
            elif tmp == 'moc':
                s += 'ơ'
                i += 1
            elif tmp == 'j':
                s += 'ọ'
                i += 1
                bf = 'ọ'
            elif tmp == 's':
                s += 'ó'
                i+= 1
            elif tmp == 'f':
                s += 'ò'
                i+= 1
            elif tmp == 'r':
                s += 'ỏ'
                i+=1 
            elif tmp == 'x':
                s += 'õ'
                i += 1
            else:
                s += 'o'
                bf = character

        elif character == 'u' and character != bf:
            tmp = str(result[i+1])
            bf = tmp
            if tmp == 'moc':
                s += 'ư'
                i += 1
            elif tmp == 'j':
                s += 'ụ'
                i += 1
                bf = 'ụ'
            elif tmp == 's':
                s += 'ú'
                i+= 1
            elif tmp == 'f':
                s += 'ù'
                i+= 1
            elif tmp == 'r':
                s += 'ủ'
                i+=1 
            elif tmp == 'x':
                s += 'ũ'
                i += 1
            else:
                s += 'u'
                bf = character  

        elif character == 'e' and character != bf:
            tmp = str(result[i+1])
            bf = tmp
            if tmp == 'non':
                s += 'ê'

                i += 1
            elif tmp == 'j':
                s += 'ẹ'
                i += 1
                bf = 'ẹ'
            elif tmp == 's':
                s += 'é'
                i+= 1
            elif tmp == 'f':
                s += 'è'
                i+= 1
            elif tmp == 'r':
                s += 'ẻ'
                i+=1 
            elif tmp == 'x':
                s += 'ẽ'
                i += 1
            else:
                s += 'e'
                bf = character
        elif character == 'i' and character != bf:
            tmp = str(result[i+1])
            bf = tmp
            if tmp == 'j':
                s += 'ị'
                i += 1
                bf = 'ị'
            elif tmp == 's':
                s += 'í'
                i+= 1
            elif tmp == 'f':
                s += 'ì'
                i+= 1
            elif tmp == 'r':
                s += 'ỉ'
                i+=1 
            elif tmp == 'x':
                s += 'ĩ'
                i += 1
            else:
                s += 'i'
                bf = character
        else:
            s += character
            bf = character
    # print(s)
    return render_template('index.html', result = s)



if __name__ == '__main__':
    app.run(debug=True)