from tensorflow.keras.models import load_model
from skimage import transform
from skimage import exposure
from flask import Flask, request
import numpy as np
import cv2
import tensorflow as tf
from object_detection.utils import label_map_util
import warnings
from matplotlib import pyplot as plt
from PIL import Image
import glob as glob
import sys
from object_detection.utils import visualization_utils as vis_util

sign_recognition_model = None
sign_recognition_label_names = None
app = Flask(__name__)


def run_sign_detection_model():
    path_to_ckpt = 'models/sign_detection/saved_model.pb'
    path_to_labels = 'models/gtsdb3_label_map.pbtxt'
    num_classes = 3

    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(path_to_ckpt, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    label_map = label_map_util.load_labelmap(path_to_labels)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=num_classes, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    print(label_map)


def load_sign_recognition_model():
    global sign_recognition_model
    global sign_recognition_label_names

    print("[INFO] loading sign recognition model...")
    sign_recognition_model = load_model("models/sign_recognition")
    sign_recognition_label_names = open("models/sign_recognition/labels/signnames.csv").read().strip().split("\n")[1:]
    sign_recognition_label_names = [l.split(",")[1] for l in sign_recognition_label_names]


@app.route('/predict', methods=['POST'])
def get_prediction():
    filestr = request.files['file'].read()  # "file" key'i ile gonderilen resmi al
    npimg = np.frombuffer(filestr, np.uint8)
    image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    curr_sign_recognition_label = process_sign_recognition(image)
    return str(curr_sign_recognition_label)


def process_sign_recognition(image):
    image = transform.resize(image, (32, 32))
    image = exposure.equalize_adapthist(image, clip_limit=0.1)

    # preprocess the image by scaling it to the range [0, 1]
    image = image.astype("float32") / 255.0
    image = np.expand_dims(image, axis=0)
    predictions = sign_recognition_model.predict(image)
    j = predictions.argmax(axis=1)[0]
    curr_label = sign_recognition_label_names[j]
    return curr_label


if __name__ == '__main__':
    # load_sign_recognition_model()  # load model at the beginning once only
    run_sign_detection_model()
    app.run(debug=True, host='0.0.0.0', port=84)
