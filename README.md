## One Bad Berry Can Spoil the Bunch: Determine the Ripeness of Strawberries using SSD and YOLO
---
### Contents
1. [Overview](#overview)
2. [Performance](#performance)
3. [Dataset](#dataset)
4. [How to use it](#how-to-use-it)

### Overview
This repo contains the dataset and pre-trained SSDMobilenetV2 and YoloV3 models to detect 3 categories of strawberries:
 - Ripe
 - Unripe (Green)
 - Rotten (Bad)

The work was done as an 8th grade science project assignment for Access Academy in Portland, OR, USA.

The YoloV3 model was trained using the scripts from https://github.com/OlafenwaMoses/ImageAI/tree/master/imageai/Detection  - special shout out to https://github.com/OlafenwaMoses for creating an easy to use framework.

The SSDMobilenetV2 model was trained according to the procedure outlined in https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html

### Performance
The table below outlines the average precision and mAP metrics for both models
<table width="70%">
  <tr>
    <td></td>
    <td colspan=3 align=center>Average Precision Per Class</td>
	<td colspan=3 align=center><i>mAP</td>
  </tr>
  <tr>
    <td></td>
    <td align=center>Ripe</td>
    <td align=center>Green</td>
    <td align=center>Bad</td>
	<td>
  </tr>
  <tr>
    <td><b>YOLOV3</td>
    <td align=center width="26%"><b>0.85</td>
    <td align=center width="26%"><b>0.74</td>
    <td align=center width="26%"><b>0.87</a></td>
	<td align=center width="26%"><b>0.82</a></td>
  </tr>
  <tr>
    <td><b>SSD MobilenetV2</td>
    <td align=center width="26%"><b>0.76</td>
    <td align=center width="26%"><b>0.69</td>
    <td align=center width="26%"><b>0.67</a></td>
	<td align=center width="26%"><b>0.70</a></td>
  </tr>
</table>

### Dataset
The dataset directory contains:
- train - 378 images and their corresponding PASCAL/VOC files used for training
- test - 134 images and their corresponding PASCAL/VOC files used for validating the models' performance
- ground_truth - the 134 test images with their ground truth bounding boxes

### How to use it
#### Pre-requisites
- Python 3.6
- Tensorflow 1.15.0
- Keras - 2.3.1
- ImageAI - https://github.com/OlafenwaMoses/ImageAI/tree/master/imageai/Detection
- Tensorflow models - https://github.com/tensorflow/models

#### YoloV3
The yolov3 directory contains the pretrained model model_berry.h5 and the detection_config.json files needed to run the yolov3_detect.py script. The yolov3_detect.py script assumes both files are in the same directory!

Usage:

```c
yolov3_detect.py <model_dir> [image_file] [threshold]
```

The detected directory contains the result of the script execution against the test set.

#### SSDMobilenetV2
The ssd_mobilenet_v2 directory contains:
 - exported_model - exported inference protobuf graph file
 - annotations - the label map protobuf file

Usage:

```c 
	infer.py [flags]
flags:

infer.py:
  --image_path: Path to image/s
    (default: '')
  --label_path: Path to the label_map.pbtxt
    (default: '')
  --model_path: Path to the exported model
    (default: '')
```

The detected directory contains the result of the script execution against the test set.