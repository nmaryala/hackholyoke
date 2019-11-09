import os
import cv2
from uuid import uuid4

from flask import Flask, request, render_template, send_from_directory

from keras.models import load_model
from sklearn.datasets import load_files   
from keras.utils import np_utils
from glob import glob
from keras import applications, callbacks
from keras.preprocessing.image import ImageDataGenerator 
from keras import optimizers
from keras.models import Sequential,Model,load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D,GlobalAveragePooling2D
from keras.callbacks import TensorBoard,ReduceLROnPlateau,ModelCheckpoint
from keras.optimizers import SGD, Adam
from PIL import Image
import numpy as np
import tensorflow as tf


__author__ = 'nmaryala'

app = Flask(__name__)
# app = Flask(__name__, static_folder="images")

global model
base_model = applications.resnet50.ResNet50(weights= None, include_top=False, input_shape= (64,64,3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
lr =  0.02535465287856519
do =  0.5762909269041454
x = Dropout(do)(x) # original = 0.7
num_classes = 10
predictions = Dense(num_classes, activation= 'softmax')(x)

x = Dropout(do)(x) # original = 0.7
predictions = Dense(num_classes, activation= 'softmax')(x)
model = Model(inputs = base_model.input, outputs = predictions)
global graph
graph = tf.get_default_graph()
model = Model(inputs = base_model.input, outputs = predictions)
best_weights = 'weights-improvement-29-0.77.hdf5'
weight_path = 'weights/{}'.format(best_weights)
model.load_weights(weight_path)
adam = Adam(lr= lr)
model.compile(optimizer= adam, loss='categorical_crossentropy', metrics=['accuracy'])
APP_ROOT = os.path.dirname(os.path.abspath(__file__))

@app.route("/")
def index():
    return render_template("website.html")

@app.route("/upload", methods=["POST"])
def upload():
    target = os.path.join(APP_ROOT, 'images/')
    # target = os.path.join(APP_ROOT, 'static/')
    print(target)
    if not os.path.isdir(target):
            os.mkdir(target)
    else:
        print("Couldn't create upload directory: {}".format(target))
    print(request.files.getlist("file"))
    for upload in request.files.getlist("file"):
        print(upload)
        print("{} is the file name".format(upload.filename))
        filename = upload.filename
        destination = "/".join([target, filename])
        print ("Accept incoming file:", filename)
        print ("Save it to:", destination)
        upload.save(destination)
        name = "images/"+upload.filename
        X_test = cv2.imread(name)
        img = Image.fromarray(np.uint8(X_test/255))
        img = img.resize((64, 64), Image.ANTIALIAS)
        
        mean = 0
        std = 1
        img = (np.array(img) - mean)/std
        diseases = 'Actinic keratoses and intraepithelial carcinoma / Bowen\'s disease (akiec), basal cell carcinoma (bcc), benign keratosis-like lesions (solar lentigines / seborrheic keratoses and lichen-planus like keratoses, bkl), dermatofibroma (df), melanoma (mel), melanocytic nevi (nv) and vascular lesions (angiomas, angiokeratomas, pyogenic granulomas and hemorrhage, vasc).'
        labels = diseases.split(',')
        print(labels)
        print(len(labels))
        dic = {}

        with graph.as_default():
            prediction = model.predict(img.reshape(1, 64, 64, 3))[0]
            print(type(prediction))
            for i in range(len(list(prediction))):
                dic[i] = prediction[i]
            dic2 = {}
            dic3 = {}
            count = 0
            maj = ''
            certain = 0
            maj1 = ''
            certain1 = 0
            maj2 = ''
            certain2 = 0
            for i in sorted(dic , key=dic.get, reverse = True)[:3]:
                dic2[i] = dic[i]
                dic3[labels[i]] = dic[i]
                if count == 0:
                    maj = labels[i]
                    certain = dic[i]
                if count == 1:
                    maj1 = labels[i]
                    certain1 = dic[i]
                if count == 2:
                    maj2 = labels[i]
                    certain2 = dic[i]

                count += 1

            print('Prediction: ', dic2)



    # return send_from_directory("images", filename, as_attachment=True)
    return render_template("report.html", image_name=filename, preds = dic3, major = maj, cert = certain, major1 = maj1, cert1 = certain1, major2 = maj2, cert2 = certain2)

@app.route('/upload/<filename>')
def send_image(filename):
    return send_from_directory("images", filename)

if __name__ == "__main__":
    app.run(port=4555, debug=True)
