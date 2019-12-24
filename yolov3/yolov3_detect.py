import cv2
import sys
import os
import shutil
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import random
import fileinput
import time
import numpy as np
import imageai
from imageai.Detection.Custom import DetectionModelTrainer
from imageai.Detection.Custom import CustomObjectDetection
from matplotlib import image as mpltimage
from matplotlib import pyplot as plt

# Location of the pre-compiled dependencies
#sys.path.append("~/code/deeplearn/tensorflow/models/research")
#sys.path.append("~/code/deeplearn/tensorflow/models/research/slim")
print(imageai.__path__)

from matplotlib import image as mpltimage
from matplotlib import pyplot as plt
from PIL import Image
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


def resize_with_aspect_ratio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)
'''
def show_image(image_np, image_path):
    plt.title(os.path.split(image_path)[1])
    imgplot=plt.imshow(image_np)
    plt.show()
'''

def from_yolo_detections_to_tensorflow_detections(detections, im_width, im_height, threshold=0.0):
    classes = np.empty(shape=(0,0))
    scores = np.empty(shape=(0,0))
    boxes = np.empty(shape=(len(detections),4))
    class_name_dict = {'Ripe Strawberry':1, 'Green Strawberry':2, 'Bad Strawberry':3}
    for obj in detections:
        classes = np.append(classes, class_name_dict[obj['name']])
        scores = np.append(scores, obj['percentage_probability']/100.0)
        box = obj['box_points']
        boxes[len(scores)-1,:] = [box[1]/im_height, box[0]/im_width, box[3]/im_height, box[2]/im_width]
    return boxes, scores, classes

def show_image(image_path, detections):
    image = Image.open(image_path)
    (im_width, im_height) = image.size
    image_np = np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)
    [boxes, scores, classes] = from_yolo_detections_to_tensorflow_detections(
        detections, im_width, im_height)
    print(boxes)
    print(scores)
    print("len(scores): " + str(len(scores)))
    cat_index = {1: {'id': 1, 'name': 'Ripe Strawberry'}, 2: {'id': 2, 'name': 'Green Strawberry'}, 3: {'id': 3, 'name': 'Bad Strawberry'}}
    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        boxes, #np.squeeze(boxes),
        (classes).astype(np.int32),
        scores,
        cat_index,
        use_normalized_coordinates=True,
        line_thickness=4,
        max_boxes_to_draw = len(scores))
    #plt.title(os.path.split(image_path)[1])
    #imgplot=plt.imshow(image_np)
    #plt.show()
    out_dir = './detected'
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    image_name = os.path.split(image_path)[1]
    new_image_path = os.path.join(out_dir, image_name)
    print("Saving detected image to: " + new_image_path)
    mpltimage.imsave(new_image_path, image_np)


def show_image_ocv(image_path, detections, threshold=0.5):
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print("Failed to open " + image_path)
        return
    
    for obj in detections:
        if obj['percentage_probability'] < threshold:
            continue
        x1,y1,x2,y2 = obj['box_points']
        obj_name = obj['name']
        if obj_name == 'Ripe Strawberry':
            color = (0,0,255)
        elif obj_name == 'Green Strawberry':
            color = (0,255,0)
        else:
            color = (255,0,0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        image_text = obj_name + ': ' + str(int(obj['percentage_probability'])) + '%'
        #print(image_text)
        cv2.putText(img, image_text, (x1,y1+10),0,0.4,color)

    img = resize_with_aspect_ratio(img, 640, 480)
    cv2.imshow(os.path.split("Detection: " + image_path)[1],img)
    cv2.waitKey()  
    cv2.destroyAllWindows()

def bb_intersection_over_union(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
 
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
 
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
 
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
 
	# return the intersection over union value
	return iou

def filter_out_overlapeed_detections(detections, iou_threshold):
    filtered_det = []
    for obj in detections:
        add = True
        for obj2 in detections:
            if obj2 != obj and bb_intersection_over_union(obj['box_points'], obj2['box_points']) > iou_threshold \
                and obj['percentage_probability'] <= obj2['percentage_probability']:
                add = False
                break
        if add:
            filtered_det.append(obj)
    return filtered_det

def yolov3_detect_objects(detector, image_path, dest_dir, iou_threshold=0.5):
    image_base_name, ext = os.path.splitext(os.path.split(image_path)[1])
    out_path = os.path.join(dest_dir, image_base_name+"_detect"+ext)
    detections = detector.detectObjectsFromImage(input_image=image_path, \
        output_image_path=out_path)
    os.remove(out_path)
    return filter_out_overlapeed_detections(detections, iou_threshold)

def yolov3_detect_objects_and_show_image(detector, image_path, dest_dir):
    print("Detecting berries in " + image_path)
    tick = time.time()
    detections = yolov3_detect_objects(detector, image_path, dest_dir)
    elapsed_time = time.time() - tick
    print("Berries detected in: " + str(elapsed_time) + " secs")
    for detection in detections:
        print(detection["name"], " : ", detection["percentage_probability"], " : ", detection["box_points"])
    show_image(image_path, detections)

def main():
    if (len(sys.argv) < 2):
        print("Usage: " + sys.argv[0] + " <model_dir> [image_file] [threshold]")
        exit(-1)

    model_dir = sys.argv[1]
    
    if (len(sys.argv) > 2):
        image_name = sys.argv[2]
    else:
        print("Enter image or directory name: ")
        image_name = input()

    dest_dir = model_dir

    threshold = 0.75
    if (len(sys.argv) > 3):
        threshold = sys.argv[3]

    detector = CustomObjectDetection()
    detector.setModelTypeAsYOLOv3()
    detector.setModelPath(os.path.join(model_dir, "model_berry.h5"))
    detector.setJsonPath(os.path.join(model_dir,"detection_config.json"))
    print("Loading model....")
    detector.loadModel()
    print("Model loaded, ready to detect me some berries!")

    if os.path.isdir(image_name):
        #it is a directory so iterate over all the images
        image_dir = image_name
        for  file_name in os.listdir(image_dir):
            image_base_name, ext = os.path.splitext(file_name)
            if ext.lower() == '.jpg' or ext.lower() == '.png' or ext.lower() == '.jpeg':
                yolov3_detect_objects_and_show_image(detector, os.path.join(image_dir, file_name), dest_dir)
            else:
                print("Skipping " + file_name)
    else:
        while len(image_name) > 0 and image_name != 'no' and image_name != 'exit':
            yolov3_detect_objects_and_show_image(detector, image_name, dest_dir)
            print("Enter new image name (no or exit to quit): ")
            image_name = input()

    print("Done!")

main()