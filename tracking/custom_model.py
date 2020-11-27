import numpy as np
import os
# import six.moves.urllib as urllib
import sys
# import tarfile
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import zipfile
import cv2
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from windowcapture import WindowCapture
from pywinauto import Desktop
from time import time
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.75)

# This is needed since the notebook is stored in the object_detection folder.
# sys.path.append("..")

# script repurposed from sentdex's edits and TensorFlow's example script. Pretty messy as not all unnecessary
# parts of the original have been removed

# # Model preparation

# ## Variables
#
# Any model exported using the `export_inference_graph.py` tool can be loaded here simply by changing `PATH_TO_CKPT` to point to a new .pb file.
#
# By default we use an "SSD with Mobilenet" model here. See the [detection model zoo](https://github.com/tensorflow/models/blob/master/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.


# What model to download.
MODEL_NAME = 'new_graph/all_with_frames'  # change to whatever folder has the new graph
# MODEL_FILE = MODEL_NAME + '.tar.gz'   # these lines not needed as we are using our own model
# DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('training', 'object-detection.pbtxt')  # our labels are in training/object-detection.pbkt

NUM_CLASSES = 18  # we only are using one class at the moment (mask at the time of edit)


# ## Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')


# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  
# Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

color = (0, 0, 255)
font = cv2.FONT_HERSHEY_SIMPLEX
org = (50, 50)
thickness = 2
fontScale = 1

windows = Desktop(backend="uia").windows()
for i in windows:
    if i.window_text().startswith("GS"): 
        windowname = i.window_text()
        print("Found Emulator")

wincap = WindowCapture(windowname)
loop_time = time()

with detection_graph.as_default():
    with tf.Session(graph=detection_graph, config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        while True:
            # for image_path in TEST_IMAGE_PATHS:
                # image = Image.open(image_path)
            image_np = wincap.get_screenshot()

            # result image with boxes and labels on it.
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            # Actual detection.
            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
            # Visualization of the results of a detection.
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                min_score_thresh=0.1,
                line_thickness=4)
        
            image_np = cv2.putText(image_np, 'FPS {}'.format(int(1 / (time() - loop_time))), org, font,  
                   fontScale, color, thickness, cv2.LINE_AA) 
            cv2.imshow('4622 Project', image_np)

            loop_time = time()
            # plt.figure(figsize=IMAGE_SIZE)
            # plt.imshow(image_np)    # matplotlib is configured for command line only so we save the outputs instead
            # plt.savefig("outputs/detection_output{}.png".format(i))  # create an outputs folder for the images to be saved
            # Press "q" to quit
            if cv2.waitKey(1) == ord("q"):
                cv2.destroyAllWindows()
                break