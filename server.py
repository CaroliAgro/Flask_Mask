from flask import Flask, request, render_template, request
from skimage.color import rgb2gray, gray2rgb
from PIL import Image
from io import BytesIO
import numpy as np
import base64
import os
import sys
from mrcnn.config import Config
from mrcnn import model as modellib, utils
import tensorflow as tf
from io import BytesIO
import skimage.draw
from skimage import io
from keras.models import load_model

from tensorflow.python.keras.backend import set_session
sess = tf.Session()
graph = tf.get_default_graph()

app = Flask(__name__)
model = None

def prepare_model():
    global model
    global sess
    global cfg
    set_session(sess)
    class CarolConfig(Config):
        NAME = "cereja"
        IMAGES_PER_GPU = 2
        NUM_CLASSES = 1 + 1  
        STEPS_PER_EPOCH = 100
        DETECTION_MIN_CONFIDENCE = 0.9
    config = CarolConfig()
    class InferenceConfig(config.__class__):
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
    config = InferenceConfig()
    MODEL_PATH = "logs/cereja20200623T0202/mask_rcnn_cereja_0005.h5"

    DEFAULT_LOGS_DIR = os.path.join("logs")
    #DEVICE = "/cpu:0" 
    #with tf.device(DEVICE):
        #_model = modellib.MaskRCNN(mode="inference", config=config, model_dir=DEFAULT_LOGS_DIR)
    model = modellib.MaskRCNN(mode="inference", config=config, model_dir=DEFAULT_LOGS_DIR)
    #_model.load_weights(MODEL_PATH, by_name=True)
    model.load_weights(MODEL_PATH, by_name=True)
    

prepare_model()

def color_splash(image, mask):
    gray = gray2rgb(rgb2gray(image)) * 255
    #c = np.zeros([image.shape[0], image.shape[1], 3])
    if mask.shape[-1] > 0:
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = c.astype(np.uint8)
    return splash

def encode(image):
    img = Image.fromarray(image.astype("uint8"))
    rawBytes = BytesIO()
    img.save(rawBytes, "JPEG")
    rawBytes.seek(0)
    img_base64 = base64.b64encode(rawBytes.getvalue()).decode('ascii')
    mime = "image/jpeg"
    uri = "data:%s;base64,%s"%(mime, img_base64)
    return uri
    
def decode():
    file = request.files['file']
    byte_image = file.stream.read()
    numpy_image = np.array(Image.open(BytesIO(byte_image)))
    return numpy_image

@app.route('/handle_form', methods=['POST'])
def handle_form():
    global sess
    global graph
    numpy_image = decode()
    #model = prepare_model()
    with graph.as_default():
        set_session(sess)
        results = model.detect([numpy_image], verbose=1)
        r = results[0]
        splash = color_splash(numpy_image, r['masks'])
        n_graos = len(r['class_ids'])
    uri_image = encode(splash)
    return render_template('index.html', image=uri_image, n= n_graos)
    

@app.route("/")
def index():
    return render_template("index.html");   

if __name__ == "__main__":
    app.run(debug=True)

#https://blog.csdn.net/qq_43381010/article/details/104576046