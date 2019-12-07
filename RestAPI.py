from flask import Flask, request, jsonify
import uuid
import os
import numpy as np
from flask import Flask, request, redirect, url_for
from werkzeug.utils import secure_filename

from io import BytesIO

from keras.models import load_model
import cv2

UPLOAD_FOLDER = '/Users/Test/'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
IMAGE_WIDTH = 200
IMAGE_HEIGHT = 200
IMAGE_CHANNELS = 3
class_labels={'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9, 'K': 10, 'L': 11, 'M': 12, 'N': 13, 'O': 14, 'P': 15, 'Q': 16, 'R': 17, 'S': 18, 'T': 19, 'U': 20, 'V': 21, 'W': 22, 'X': 23, 'Y': 24, 'Z': 25, 'nothing': 26, 'space': 27}
labels=list(class_labels.keys())
model = load_model('FinalModel.h5')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER



@app.route('/api/image', methods=['POST'])
def upload_image():
  # check if the post request has the file part
  if 'image' not in request.files:
      return jsonify({'error':'No posted image. Should be attribute named image.'})
  file = request.files['image']

  # if user does not select file, browser also
  # submit a empty part without filename
  if file.filename == '':
      return jsonify({'error':'Empty filename submitted.'})
  if file and allowed_file(file.filename):
      filename = secure_filename(file.filename)
      print("***2:"+filename)
      #file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
      x = []
      ImageFile.LOAD_TRUNCATED_IMAGES = False
      img=cv2.imread(BytesIO(file.read()),0)
      img_gray = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
      img_gray=cv2.resize(img_gray,(200,200))
      img_gray=np.reshape(img_gray,[1,200,200,1])
      ar=model.predict(img_gray)
      ind=np.argmax(ar[0])
      lst = decode_predictions(pred, top=5)
      response = {'Sign predicted':labels[ind],'Accuracy':str(int(ar[0][ind])*100)+"%"}
      return jsonify(response)
  else:
      return jsonify({'error':'File has invalid extension'})

if __name__ == '__main__':
    app.run(port=5000,debug=True)