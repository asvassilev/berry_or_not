import json
from json import encoder
import os
import sys

# Location of the pre-compiled dependencies
sys.path.append("~/code/deeplearn/tensorflow/models/research")

# Now that the script knows where to look, we can safely import our objects
#import cv2
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from PIL import Image
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from IPython.display import display

flags = tf.app.flags
flags.DEFINE_string('model_path', '', 'Path to the exported model')
flags.DEFINE_string('image_path', '', 'Path to image/s')
flags.DEFINE_string('label_path', '', 'Path to the label_map.pbtxt')
FLAGS = flags.FLAGS

def get_category_index():
    # Loading label map
    label_map = label_map_util.load_labelmap(FLAGS.label_path)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,                                                        use_display_name=True)
    return label_map_util.create_category_index(categories)


def detect_objects(image_np, sess, detection_graph, cat_index):
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # Each box represents a part of the image where a particular object was detected.
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    # Each score represent the level of confidence for each of the objects.
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
        cat_index,
        use_normalized_coordinates=True,
        line_thickness=4)
    
    #Image.fromarray(image_np).show()
    
    return scores, classes, image_np

def show_image(image_np, image_path):
    plt.title(os.path.split(image_path)[1])
    imgplot=plt.imshow(image_np)
    plt.show()

def detect_and_show_image(image_path, sess, detection_graph, cat_index):
    image = Image.open(image_path)
    (im_width, im_height) = image.size
    image_np = np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)
    scores, classes, image_with_labels = detect_objects(image_np, sess, detection_graph, cat_index)
    show_image(image_with_labels, image_path)

def main(_):
    
    # Load a (frozen) Tensorflow model into memory.
    cat_index = get_category_index()
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        path_to_ckpt = os.path.join(FLAGS.model_path, 'frozen_inference_graph.pb')
        print("Model path: " + path_to_ckpt)
        with tf.gfile.GFile(path_to_ckpt, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        sess = tf.Session(graph=detection_graph)

    img_path = FLAGS.image_path
    if os.path.isdir(img_path):
        for f in os.listdir(img_path):
            image_base_name, ext = os.path.splitext(f)
            if ext.lower() == '.jpg' or ext.lower() == '.png' or ext.lower() == '.jpeg':
                detect_and_show_image(os.path.join(img_path, f), sess, detection_graph, cat_index)
    else:
        detect_and_show_image(img_path, sess, detection_graph, cat_index)
        #scores, classes, image_with_labels = detect_objects(image_np, sess, detection_graph)
        #show_image(image_with_labels)

    sess.close()

if __name__ == '__main__':
    tf.app.run()