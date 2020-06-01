from models.lane_detection.calibration_utils import calibrate_camera, undistort
from models.lane_detection.binarization_utils import binarize
from models.lane_detection.perspective_utils import birdeye
from models.lane_detection.line_utils import get_fits_by_sliding_windows, Line, get_fits_by_previous_fits
from models.lane_detection.globals import xm_per_pix, time_window

from flask import Flask, request, jsonify
import numpy as np
import cv2

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
#from skimage.color import rgb2gray
from scipy.ndimage import gaussian_filter
from scipy import signal
from PIL import Image
import numpy as np
import sys
from collections import namedtuple
import math
import cv2
from matplotlib import cm


processed_frames = 0                    # counter of frames processed (when processing video)
line_lt = Line(buffer_len=time_window)  # line on the left of the lane
line_rt = Line(buffer_len=time_window)  # line on the right of the lane
sign_recognition_model = None
sign_recognition_label_names = None
app = Flask(__name__)


def prepare_out_blend_frame(blend_on_road, img_binary, img_birdeye, img_fit, line_lt, line_rt, offset_meter):
    h, w = blend_on_road.shape[:2]

    thumb_ratio = 0.2
    thumb_h, thumb_w = int(thumb_ratio * h), int(thumb_ratio * w)

    off_x, off_y = 20, 15

    # add a gray rectangle to highlight the upper area
    mask = blend_on_road.copy()
    mask = cv2.rectangle(mask, pt1=(0, 0), pt2=(w, thumb_h+2*off_y), color=(0, 0, 0), thickness=cv2.FILLED)
    blend_on_road = cv2.addWeighted(src1=mask, alpha=0.2, src2=blend_on_road, beta=0.8, gamma=0)

    # add thumbnail of binary image
    thumb_binary = cv2.resize(img_binary, dsize=(thumb_w, thumb_h))
    thumb_binary = np.dstack([thumb_binary, thumb_binary, thumb_binary]) * 255
    blend_on_road[off_y:thumb_h+off_y, off_x:off_x+thumb_w, :] = thumb_binary

    # add thumbnail of bird's eye view
    thumb_birdeye = cv2.resize(img_birdeye, dsize=(thumb_w, thumb_h))
    thumb_birdeye = np.dstack([thumb_birdeye, thumb_birdeye, thumb_birdeye]) * 255
    blend_on_road[off_y:thumb_h+off_y, 2*off_x+thumb_w:2*(off_x+thumb_w), :] = thumb_birdeye

    # add thumbnail of bird's eye view (lane-line highlighted)
    thumb_img_fit = cv2.resize(img_fit, dsize=(thumb_w, thumb_h))
    blend_on_road[off_y:thumb_h+off_y, 3*off_x+2*thumb_w:3*(off_x+thumb_w), :] = thumb_img_fit

    # add text (curvature and offset info) on the upper right of the blend
    mean_curvature_meter = np.mean([line_lt.curvature_meter, line_rt.curvature_meter])
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(blend_on_road, 'Curvature radius: {:.02f}m'.format(mean_curvature_meter), (860, 60), font, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(blend_on_road, 'Offset from center: {:.02f}m'.format(offset_meter), (860, 130), font, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

    return blend_on_road


def compute_offset_from_center(line_lt, line_rt, frame_width):
    if line_lt.detected and line_rt.detected:
        line_lt_bottom = np.mean(line_lt.all_x[line_lt.all_y > 0.95 * line_lt.all_y.max()])
        line_rt_bottom = np.mean(line_rt.all_x[line_rt.all_y > 0.95 * line_rt.all_y.max()])
        lane_width = line_rt_bottom - line_lt_bottom
        midpoint = frame_width / 2
        offset_pix = abs((line_lt_bottom + lane_width / 2) - midpoint)
        offset_meter = xm_per_pix * offset_pix
    else:
        offset_meter = -1

    return offset_meter


def process_pipeline(frame, keep_state=True):
    global line_lt, line_rt, processed_frames

    # undistort the image using coefficients found in calibration
    img_undistorted = undistort(frame, mtx, dist, verbose=False)

    # binarize the frame s.t. lane lines are highlighted as much as possible
    img_binary = binarize(img_undistorted, verbose=False)

    # compute perspective transform to obtain bird's eye view
    img_birdeye, M, Minv = birdeye(img_binary, verbose=False)

    # fit 2-degree polynomial curve onto lane lines found
    if processed_frames > 0 and keep_state and line_lt.detected and line_rt.detected:
        line_lt, line_rt, img_fit = get_fits_by_previous_fits(img_birdeye, line_lt, line_rt, verbose=False)
    else:
        line_lt, line_rt, img_fit = get_fits_by_sliding_windows(img_birdeye, line_lt, line_rt, n_windows=9, verbose=False)

    # compute offset in meter from center of the lane
    offset_meter = compute_offset_from_center(line_lt, line_rt, frame_width=frame.shape[1])
    processed_frames += 1

    print('OFFSET:', offset_meter)

    return offset_meter


def hough_transform(edge_img, no_of_lines):
    # Build a dictionary to accumulate the votes. The keys are named tuples representing lines and the values are the votes that
    #  those lines receive.
    Line = namedtuple("Line", ["ro", "theta"])
    votes = {}

    # Since theta can have infinietly many different values, I discretized pi radians into 180 values.
    # The line proposals will be made with these 60 theta values.
    no_of_quantiles = 180
    thetas = np.linspace(0, math.pi, num=no_of_quantiles, retstep=True)

    # Iterate over the edge image and for each edge pixel, try line proposals and record the lines that such point voted for
    for row_idx, row in enumerate(edge_img):
        for col_idx, col in enumerate(row):
            if col == 255:  # The canny edge detector sets the value for edge pixels and 0 for the rest
                for theta in thetas[0]:
                    # Calculate the distance value
                    ro = int(round(col_idx * math.cos(theta) + row_idx * math.sin(theta)))
                    line = Line(ro=ro, theta=theta)

                    # If the line exists in the dictionary, incerement its vote by 1 else add it to the dictionary and set
                    # its vote to 1
                    if line in votes:
                        votes[line] = votes[line] + 1
                    else:
                        votes[line] = 1

    # Sort the lines based on their votes in order to output the largest no_of_lines amount of them
    votes = {k: v for k, v in sorted(votes.items(), key=lambda item: item[1])}

    # Code segment to plot hough space (should increase the number of theta quantiles for better plotting)
    '''
    max_vote = list(votes.items())[-1][1]
    min_rho = sys.maxsize
    max_rho = -sys.maxsize
    for line in votes.keys():
        if line.ro < min_rho:
            min_rho = line.ro
        if line.ro > max_rho:
            max_rho = line.ro
    hough_space = np.zeros([max_rho - min_rho, no_of_quantiles])
    for line in votes.keys():
        hough_space[line.ro - min_rho - 1][int(round(line.theta * no_of_quantiles / (math.pi))) - 1] = votes[line] * 255 / max_vote
    plt.imshow(hough_space)
    im = Image.fromarray(hough_space)
    im = im.convert("L")
    #im.save('PATH/TO/OUTPUT.png')
    '''

    return dict(list(votes.items())[-no_of_lines:])


def intersection(line1, line2):
    """Finds the intersection of two lines given in Hesse normal form.

    Returns closest integer pixel locations.
    See https://stackoverflow.com/a/383527/5087436
    """
    rho1, theta1 = line1
    rho2, theta2 = line2
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([[rho1], [rho2]])
    x0, y0 = np.linalg.solve(A, b)
    x0, y0 = int(np.round(x0)), int(np.round(y0))
    return [[x0, y0]]


@app.route('/predict', methods=['POST'])
def get_prediction():
    distance_from_center_arr = []
    for key in request.files:
        filestr = request.files[key].read()  # "file" key'i ile gonderilen resmi al
        # npimg = np.frombuffer(filestr, np.uint8)
        img = cv2.imread(filestr, 0)
        blurred = gaussian_filter(img, sigma=3.0)
        canny_edges = cv2.Canny(blurred, 50, 120)

        detected_lines = hough_transform(canny_edges, 15)
        lines = []
        for line in detected_lines.keys():
            lines.append([line.ro, line.theta])

        max_theta = -1000
        min_theta = 1000
        for line in lines:
            if max_theta < line[1]:
                max_theta = line[1]
                max_line = line
            if min_theta > line[1]:
                min_theta = line[1]
                min_line = line

        lines = []
        lines.append(min_line)
        lines.append(max_line)

        x, y = intersection(min_line, max_line)[0]

        print('x: ' + str(x) + ' y: ' + str(y))

        # Plot the fitted lines over the edge image
        fig, ax = plt.subplots()
        plt.ylim(canny_edges.shape[0], 0)
        plt.xlim(0, canny_edges.shape[1])
        x_vals = np.arange(canny_edges.shape[1])
        y_vals = [(line[0] / (math.sin(line[1]) + 0.00001) - x * math.tan(line[1])) for x in x_vals]
        ax.imshow(img, cmap=cm.gray)
        origin = np.array((0, canny_edges.shape[1]))
        for line in lines:
            angle = line[1]
            dist = line[0]
            y0, y1 = (dist - origin * np.cos(angle)) / (np.sin(angle) + 0.00001)
            ax.plot(origin, (y0, y1), '-r')

        ax.savefig('./img.png')
        print(lines)


        # image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        # distance_from_center = process_pipeline(img, keep_state=False)
        # distance_from_center_arr.append(distance_from_center)

    return jsonify(distance_from_center_arr="distance_from_center_arr")


if __name__ == '__main__':
    ret, mtx, dist, rvecs, tvecs = calibrate_camera(calib_images_dir='camera_cal')

    app.run(debug=True, host='0.0.0.0', port=8000)