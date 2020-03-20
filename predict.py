from tensorflow.keras.models import load_model
from skimage import transform
from skimage import exposure
from flask import Flask, request, jsonify
import numpy as np
import cv2

import json
import tensorflow as tf
import glog as log
from models.lane_detection.lanenet_model import lanenet_postprocess, lanenet, global_config

CFG = global_config.cfg
sign_recognition_model = None
sign_recognition_label_names = None
app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def get_prediction():
    filestr = request.files['file'].read()  # "file" key'i ile gonderilen resmi al
    npimg = np.frombuffer(filestr, np.uint8)
    image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # curr_sign_recognition_label = process_sign_recognition(image)
    curr_sign_recognition_label = ''
    lane_image = process_lane_recognition_model(image)
    return jsonify(sign=curr_sign_recognition_label, lane=lane_image)


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


def process_lane_recognition_model(image):

    weights_path="models/lane_detection/downloaded_model/tusimple_lanenet_vgg.ckpt"
#preprocess
    image_vis = image
    image = cv2.resize(image, (512, 256), interpolation=cv2.INTER_LINEAR)
    image = image / 127.5 - 1.0

    log.info('Image load complete')
    #model run
    tf.compat.v1.disable_eager_execution()
    input_tensor = tf.compat.v1.placeholder(dtype=tf.float32, shape=[1, 256, 512, 3], name='input_tensor')

    net = lanenet.LaneNet(phase='test', net_flag='vgg')
    binary_seg_ret, instance_seg_ret = net.inference(input_tensor=input_tensor, name='lanenet_model')

    postprocessor = lanenet_postprocess.LaneNetPostProcessor()

    saver = tf.compat.v1.train.Saver()

    # Set sess configuration
    sess_config = tf.compat.v1.ConfigProto()
    sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TEST.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = CFG.TRAIN.TF_ALLOW_GROWTH
    sess_config.gpu_options.allocator_type = 'BFC'

    sess = tf.compat.v1.Session(config=sess_config)

    with sess.as_default():
        saver.restore(sess=sess, save_path=weights_path)
        binary_seg_image, instance_seg_image = sess.run(
            [binary_seg_ret, instance_seg_ret],
            feed_dict={input_tensor: [image]}
        )
        postprocess_result = postprocessor.postprocess(
            binary_seg_result=binary_seg_image[0],
            instance_seg_result=instance_seg_image[0],
            source_image=image_vis
        )
        mask_image = postprocess_result['mask_image']

        for i in range(CFG.TRAIN.EMBEDDING_FEATS_DIMS):
            instance_seg_image[0][:, :, i] = minmax_scale(instance_seg_image[0][:, :, i])
        embedding_image = np.array(instance_seg_image[0], np.uint8)

    lists = mask_image[:, :, (2, 1, 0)].tolist()
    json_str = json.dumps(lists)
   # encoded_img = base64.encodebytes(img_byte_arr.getvalue()).decode('ascii')
    sess.close()
    return json_str


def minmax_scale(input_arr):
    min_val = np.min(input_arr)
    max_val = np.max(input_arr)
    output_arr = (input_arr - min_val) * 255.0 / (max_val - min_val)

    return output_arr


if __name__ == '__main__':
    # load model at the beginning once only
    load_sign_recognition_model()
    app.run(debug=True, host='0.0.0.0', port=80)
