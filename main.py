import os
from PIL import Image, ImageOps
from flask import Flask, render_template, request, jsonify
import base64, json
from io import BytesIO

# all ML stuff
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# initialize flask application 
app = Flask(__name__)
HOST = '0.0.0.0'
PORT = 8888

# upload the model. 
#model = tf.keras.models.load_model('static/hand_writing_model_shallow')
model = tf.keras.models.load_model('static/handwriting_model_convnet')

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    results = {"prediction" : "Empty", "probability" : {}}
    # Get data stream
    input = BytesIO(base64.urlsafe_b64decode(request.form['img']))
    input_img = load_img(input, target_size = (28, 28), color_mode="grayscale") # convert data to object PIL.Image.Image
    invert_img =ImageOps.invert(input_img) # invert black(originally 0) to whites (originally 255) 
    y = img_to_array(invert_img)/255
    y = np.expand_dims(y, axis=0) # the model requires input's dimensions = (1,28,28,1)
    images = np.vstack([y])
    classes = model.predict(images)
    results["prediction"] = str(np.argmax(classes))
    results["probability"] = float(np.max(classes)) * 100
    return json.dumps(results)



if __name__=='__main__':
    app.run(host=HOST,
            debug=True,
            port=PORT)