#Import necessary libraries
from flask import Flask, render_template, Response
import numpy as np
from keras.models import model_from_json
import operator
import cv2
import simplejpeg
import sys, os

SHRINK_RATIO = 0.25
FPS = 5
FRAME_RATE = 5
WIDTH = 640
HEIGHT = 480


# Loading the model
json_file = open("model-bw-final1.json", "r")
model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(model_json)
loaded_model.load_weights("model-bw-final1.h5")
# load weights into new model

json_file = open("model-bw_dru.json", "r")
model_json_dru = json_file.read()
json_file.close()
loaded_model_dru = model_from_json(model_json_dru)
# load weights into new model
loaded_model_dru.load_weights("model-bw_dru.h5")

json_file = open("model-bw_tkdi.json", "r")
model_json_tkdi = json_file.read()
json_file.close()
loaded_model_tkdi = model_from_json(model_json_tkdi)
# load weights into new model
loaded_model_tkdi.load_weights("model-bw_tkdi.h5")

json_file = open("model-bw_smn.json", "r")
model_json_smn = json_file.read()
json_file.close()
loaded_model_smn = model_from_json(model_json_smn)
# load weights into new model
loaded_model_smn.load_weights("model-bw_smn.h5")
classes_list = {'0': 0, 'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'H': 8, 'I': 9, 'J': 10, 'K': 11, 'L': 12, 'M': 13, 'N': 14, 'O': 15, 'P': 16, 'Q': 17, 'R': 18, 'S': 19, 'T': 20, 'U': 21, 'V': 22, 'W': 23, 'X': 24, 'Y': 25, 'Z': 26}
#categories = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'DEL', 5: 'E', 6: 'F', 7: 'G', 8: 'H',9: 'I', 10: 'J', 11: 'K', 12: 'L', 13: 'M', 14: 'N', 15: 'NOTHING', 16: 'O', 17: 'P'
                #,18: 'Q', 19: 'R', 20: 'S', 21: 'SPACE', 22: 'T', 23: 'U', 24: 'V', 25: 'W', 26: 'X', 27: 'Y', 28: 'Z'}

categories = {v: k for k, v in classes_list.items()}

#Initialize the Flask app
app = Flask(__name__)

camera = cv2.VideoCapture(0)

camera.set(3,WIDTH)
camera.set(4,HEIGHT)
camera.set(5,FPS)
camera.set(7,FRAME_RATE)




@app.route('/')
def webpage():
    return render_template("webpage.html")

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def gen_frames():  
    prev = '0'
    flag = 0
    count_for_letter = 0
    text = ""



    
    while True:

        _, frame = camera.read()
    
        frame = cv2.flip(frame, 1)
    
        x1 = int(0.5*frame.shape[1])
        y1 = 10
        x2 = frame.shape[1]-10
        y2 = int(0.5*frame.shape[1])
    
        cv2.rectangle(frame, (220-1, 9), (620+1, 419), (255,0,0) ,1)

        roi = frame[10:410, 220:520]

    
        roi = cv2.resize(roi, None, fx= SHRINK_RATIO, fy=SHRINK_RATIO)
        #cv2.imshow("Frame", frame)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
        blur = cv2.GaussianBlur(gray,(5,5),2)

    
        th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
        ret, test_image = cv2.threshold(th3, 70, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        test_image = cv2.resize(test_image, (128, 128))
    
        result = loaded_model.predict(test_image.reshape(1, 128, 128, 1))
        result_dru = loaded_model_dru.predict(test_image.reshape(1 , 128 , 128 , 1))
        result_tkdi = loaded_model_tkdi.predict(test_image.reshape(1 , 128 , 128 , 1))
        result_smn = loaded_model_smn.predict(test_image.reshape(1 , 128 , 128 , 1))

        prediction =  {'0': 0, 'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'H': 8, 'I': 9, 'J': 10, 'K': 11, 'L': 12, 'M': 13, 'N': 14, 'O': 15, 'P': 16, 'Q': 17, 'R': 18, 'S': 19, 'T': 20, 'U': 21, 'V': 22, 'W': 23, 'X': 24, 'Y': 25, 'Z': 26}
        for x in prediction:
            prediction[x] = result[0][prediction[x]]
    
        prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
        current_symbol = prediction[0][0]

    
        if(current_symbol == 'D' or current_symbol == 'R' or current_symbol == 'U'):
                pred = {}
                pred['D'] = result_dru[0][0]
                pred['R'] = result_dru[0][1]
                pred['U'] = result_dru[0][2]
                pred = sorted(pred.items(), key=operator.itemgetter(1), reverse=True)
                current_symbol = pred[0][0]
            
    
        elif(current_symbol == 'D' or current_symbol == 'I' or current_symbol == 'K' or current_symbol == 'T'):
                prediction = {}
                prediction['D'] = result_tkdi[0][0]
                prediction['I'] = result_tkdi[0][1]
                prediction['K'] = result_tkdi[0][2]
                prediction['T'] = result_tkdi[0][3]
                prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
                current_symbol = prediction[0][0]

        elif(current_symbol == 'M' or current_symbol == 'N' or current_symbol == 'S'):
                prediction1 = {}
                prediction1['M'] = result_smn[0][0]
                prediction1['N'] = result_smn[0][1]
                prediction1['S'] = result_smn[0][2]
                prediction1 = sorted(prediction1.items(), key=operator.itemgetter(1), reverse=True)
                if(prediction1[0][0] == 'S'):
                    current_symbol = prediction1[0][0]
                else:
                    current_symbol = prediction[0][0]
                
        

        if current_symbol == '0':
            current_symbol = 'nothing'

        

        if(prev == current_symbol and current_symbol != 'nothing'):

            count_for_letter += 1

        else:
            count_for_letter = 0
            flag = 0

        prev = current_symbol
        
        if(count_for_letter >= 10 and flag != 1):
            text = text + current_symbol
            flag = 1
        
        cv2.putText(frame, "Word : " + text, (10, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,255), 4)   
        cv2.putText(frame,"Current alphabet : " + current_symbol, (10, 100), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,255), 4)   
        if not _:
            break
        else:
            ret = simplejpeg.encode_jpeg(frame, colorspace='bgr')
            #frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + ret + b'\r\n')  # concat frame one by one and show result
        #interrupt = cv2.waitKey(10)
        #if interrupt & 0xFF == 27: # esc key
            #break
        
 
#camera.release()
#cv2.destroyAllWindows()


if __name__ == "__main__":
    app.run(debug=True)