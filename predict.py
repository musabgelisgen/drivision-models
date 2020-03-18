from tensorflow.keras.models import load_model
from skimage import transform
from skimage import exposure
from flask import Flask, request
import numpy as np
import cv2

sign_recognition_model = None
sign_recognition_label_names = None
app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def get_prediction():
    filestr = request.files['file'].read()  # "file" key'i ile gonderilen resmi al
    npimg = np.frombuffer(filestr, np.uint8)
    image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    curr_sign_recognition_label = process_sign_recognition(image)
    return str(curr_sign_recognition_label)


def load_sign_recognition_model():
    global sign_recognition_model
    global sign_recognition_label_names

    print("[INFO] loading sign recognition model...")
    sign_recognition_model = load_model("models/sign_recognition")
    sign_recognition_label_names = open("models/sign_recognition/labels/signnames.csv").read().strip().split("\n")[1:]
    sign_recognition_label_names = [l.split(",")[1] for l in sign_recognition_label_names]


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
    # load model at the beginning once only
    load_sign_recognition_model()
    app.run(debug=True, host='0.0.0.0', port=80)
