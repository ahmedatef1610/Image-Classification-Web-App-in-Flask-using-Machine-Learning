from flask import Flask, render_template
from flask import request
from flask import redirect, url_for

import os
import pickle
import numpy as np
import pandas as pd
import scipy

import sklearn

import skimage
import skimage.color
import skimage.transform
import skimage.feature
import skimage.io

from termcolor import colored, cprint
# cprint(text, color=None, on_color=None, attrs=None, **kwargs)
text_color= 'blue'
bg_color= 'on_yellow'
#########################################################
app = Flask(__name__)
#########################################################
# print('__file__:    ', __file__)
# print('getcwd:      ', os.getcwd())
# print('basename:    ', os.path.basename(__file__))
# print('dirname:     ', os.path.dirname(__file__))
# print('abspath:     ', os.path.abspath(__file__))
# print('abs dirname: ', os.path.dirname(os.path.abspath(__file__)))
# print("basedir:      " ,os.path.abspath(os.path.dirname(__file__)))
#########################################################
# BASE_PATH = os.getcwd()
BASE_PATH = os.path.dirname(__file__)
UPLOAD_PATH = os.path.join(BASE_PATH, 'static/upload/')
MODEL_PATH = os.path.join(BASE_PATH, 'static/models/')
# E:\UDEMY_3\Computer Vision && Image Processing\LAB\venv\8-CV Flask Web App\3-flask\static\models\MLPClassifierModel_model_best.pickle
# -------------------- Load Models -------------------
model_sgd_path = os.path.join(MODEL_PATH, 'MLPClassifierModel_model_best.pickle')
scaler_path = os.path.join(MODEL_PATH, 'dsa_scaler.pickle')

model_sgd = pickle.load(open(model_sgd_path, 'rb'))
scaler = pickle.load(open(scaler_path, 'rb'))
#########################################################


@app.errorhandler(404)
def error404(error):
    message = "ERROR 404 OCCURRED. Page Not Found. Please go the home page and try again"
    return render_template("error.html", message=message), 404


@app.errorhandler(405)
def error405(error):
    message = 'Error 405, Method Not Found'
    return render_template("error.html", message=message), 405


@app.errorhandler(500)
def error500(error):
    message = 'INTERNAL ERROR 500, Error occurs in the program'
    return render_template("error.html", message=message), 500

#########################################################


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == "POST":
        upload_file = request.files['image_name']
        filename = upload_file.filename
        # print('The filename that has been uploaded =', filename)
        cprint(f'The filename that has been uploaded = {filename}', text_color, bg_color)
        
        # know the extension of filename
        # all only .jpg, .png, .jpeg, PNG
        ext = filename.split('.')[-1]
        # print('The extension of the filename =', ext)
        cprint(f'The extension of the filename = {ext}', text_color, bg_color)

        if ext.lower() in ['png', 'jpg', 'jpeg']:
            # saving the image
            path_save = os.path.join(UPLOAD_PATH, filename)
            upload_file.save(path_save)
            # print('File saved successfully')
            cprint(f'File saved successfully', text_color, bg_color)
            # send to pipeline model
            results = pipeline_model(path_save, scaler, model_sgd)
            hei = getheight(path_save)
            # print(results)
            cprint(f'{results}', text_color, bg_color)
            return render_template('upload.html', fileupload=True, extension=False, data=results, image_filename=filename, height=hei)
        else:
            # print('Use only the extension with .jpg, .png, .jpeg')
            cprint(f'Use only the extension with .jpg, .png, .jpeg', text_color, bg_color)
            return render_template('upload.html', fileupload=False, extension=True)
    else:
        return render_template('upload.html', fileupload=False, extension=False)


@app.route('/about/')
def about():
    return render_template('about.html')

#########################################################


def getheight(path):
    img = skimage.io.imread(path)
    h, w, _ = img.shape
    aspect = h/w
    given_width = 300
    height = given_width*aspect
    return height


def pipeline_model(path, scaler_transform, model_sgd):
    # pipeline model
    image = skimage.io.imread(path)
    # transform image into 80 x 80
    image_resize = skimage.transform.resize(image, (80, 80))
    image_scale = 255*image_resize
    image_transform = image_scale.astype(np.uint8)
    # rgb to gray
    gray = skimage.color.rgb2gray(image_transform)
    # hog feature
    feature_vector = skimage.feature.hog(gray,
                                         orientations=9,
                                         pixels_per_cell=(8, 8),
                                         cells_per_block=(3, 3))
    # scaling

    scalex = scaler_transform.transform(feature_vector.reshape(1, -1))
    result = model_sgd.predict(scalex)
    # decision function # confidence
    # decision_value = model_sgd.decision_function(scalex).flatten()
    decision_value = model_sgd.predict_proba(scalex).flatten()
    labels = model_sgd.classes_
    # probability
    z = scipy.stats.zscore(decision_value)
    prob_value = scipy.special.softmax(z)

    # top 5
    top_5_prob_ind = prob_value.argsort()[::-1][:5]
    top_labels = labels[top_5_prob_ind]
    top_prob = prob_value[top_5_prob_ind]

    # put in dictionary
    top_dict = dict()
    for key, val in zip(top_labels, top_prob):
        top_dict.update({key: np.round(val, 3)})

    return top_dict


#########################################################


if __name__ == "__main__":
    app.run(debug=True, port=8080)
