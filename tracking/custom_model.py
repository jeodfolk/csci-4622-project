import torchvision.transforms as transforms
import torch
import numpy as np
import os
import sys
import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
import cv2
from collections import defaultdict
from io import StringIO
from PIL import Image
from windowcapture import WindowCapture
from pywinauto import Desktop
from time import time
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from move_classifier import MoveClassifier
from os import path
import glob
import pprint

# adjust portion of gpu memory to used by the classifier
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)

MODEL_NAME = 'new_graph/all_auto_2'  # change to whatever folder has the new graph

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('training', 'object-detection.pbtxt')  # our labels are in training/object-detection.pbkt

NUM_CLASSES = 19

# ## Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MoveClassifier(69, device).to(device)
model.load(os.getcwd() + '/saved_models/2020_12_06_21_42_06')

# ## Loading label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

color = (0, 0, 255)
font = cv2.FONT_HERSHEY_SIMPLEX
org = (50, 50)
thickness = 2
fontScale = 1

iterr = 0
img_iter = 0
all_iter = 0

#get emulator window handle
windows = Desktop(backend="uia").windows()
for i in windows:
    if i.window_text().startswith("GS"): 
        windowname = i.window_text()
        print("Found Emulator")

save_path = "C:/Users/joe/Documents/Visual Code/code/4622/tracking/models/research/object_detection/frames"

wincap = WindowCapture(windowname)
loop_time = time()

frame_dict = {k: [] for k in range(6)} #one key for each frame needed by the move classifier
move_class = False

with detection_graph.as_default():
    with tf.Session(graph=detection_graph, config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        while True:
            # for image_path in TEST_IMAGE_PATHS:
            # image_np = cv2.imread(image_path)
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
                category_index=category_index,
                max_boxes_to_draw=2,
                use_normalized_coordinates=True,
                min_score_thresh=0.50,
                line_thickness=4)
            
            #crop image to just the bounded character
            im_height, im_width, _ = image_np.shape
            ymin = boxes[0][0][0]*im_height
            xmin = im_width - boxes[0][0][1]*im_width
            ymax = boxes[0][0][2]*im_height
            xmax = im_width - boxes[0][0][3]*im_width
            detected_char =  image_np[int(ymin):int(ymax), int(im_width-xmin):int(im_width-xmax)]
            
            #save img in a folder while keeping track of their order captured using frame_dict.
            # if(xmax > 200):
            iterr += 1
            if iterr == 4:
                save = save_path+'/{}.png'.format(str(all_iter))
                cv2.imwrite(save, detected_char)
                frame_dict[img_iter].append(save)
                img_iter += 1
                iterr = 0
                all_iter += 1
                if img_iter == 6: img_iter = 0

            # pprint.pprint(frame_dict)
            #FPS counter
            # image_np = cv2.putText(image_np, 'FPS {}'.format(int(1 / (time() - loop_time))), org, font,  
            #        fontScale, color, thickness, cv2.LINE_AA) 
            
            # if move_class and iterr == 0:
            frames = [preprocess(Image.open(f[-1]).convert('RGB')) for f in list(frame_dict.values())]
            X = torch.stack(frames)
            X = torch.unsqueeze(X, 0)
            X = X.to(device)
            # print(frames[0])
            model.eval()
            y_hat = model(X)
            pred_move = model.idx_to_move[torch.argmax(y_hat).item()]
            print(pred_move)

            cv2.imshow('4622 Project', image_np)

            # cv2.imshow('box', detected_char)

            if len(frame_dict[5]) > 0: move_class = True

            loop_time = time()

            # Press "q" to quit
            if cv2.waitKey(1) == ord("q"):
                cv2.destroyAllWindows()
                break
