"""
March 4, 2019
	author: 
		Caio E. C. Oliveira, github.com/caduunb
	usage:
		python create_tfRecordFile.py --output_path example.record
	purpose: 
		Create .record file to feed as input for training an object detection Neural Network model.
	version: 
		Python 3.6.7, tensorflow 1.12
"""

import tensorflow as tf
import yaml
import os
from object_detection.utils import dataset_util

# Define constants
flags = tf.app.flags
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
FLAGS = flags.FLAGS
INPUT_YAML = "/home/user/ObjectDetection_Model/example.yaml" 

LABELS_ID = {
  "macula" : 1
}

# Creates the TFRecord file for each image.
def create_tf_example(data_and_label_info):
  IMG_W  = 2592                           # Image width, set by the user
  IMG_H = 1944                           # Image height, set by the user
  filename = data_and_label_info["path"]  # Loads image path from the .yaml file
  filename = filename.encode()            # Encodes the file into bytes

  with tf.gfile.GFile(data_and_label_info["path"], 'rb') as fid:
    encoded_image_data = fid.read()
  image_format = b'jpeg'    # Encodes data into bytes. 
                            # Can be done using 'jpeg'.encode().
  # Lists
  xmins = []                # Normalized left x coordinates in bounding box
  xmaxs = []                # Normalized right x coordinates in bounding box
  ymins = []                # Normalized top y coordinates in bounding box
  ymaxs = []                # Normalized bottom y coordinates in bounding box
  classes_text = []         # String class name of bounding box
  classes = []              # Integer class id of bounding box

  # Feeding the lists with data from .yaml file
  for box in data_and_label_info["boxes"]:
      xmins.append(float(box['x_min'] / IMG_W))
      xmaxs.append(float(box['x_max'] / IMG_W))
      ymins.append(float(box['y_min'] / IMG_H))
      ymaxs.append(float(box['y_max'] / IMG_H))
      classes_text.append(box['label'].encode())
      classes.append(int(LABELS_ID[box['label']]))

  tf_example = tf.train.Example(
    features=tf.train.Features(
      feature={
      'image/height':             dataset_util.int64_feature(IMG_H),
      'image/width':              dataset_util.int64_feature(IMG_W),
      'image/filename':           dataset_util.bytes_feature(filename),
      'image/source_id':          dataset_util.bytes_feature(filename),
      'image/encoded':            dataset_util.bytes_feature(encoded_image_data),
      'image/format':             dataset_util.bytes_feature(image_format),
      'image/object/bbox/xmin':   dataset_util.float_list_feature(xmins),
      'image/object/bbox/xmax':   dataset_util.float_list_feature(xmaxs),
      'image/object/bbox/ymin':   dataset_util.float_list_feature(ymins),
      'image/object/bbox/ymax':   dataset_util.float_list_feature(ymaxs),
      'image/object/class/text':  dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
    }
  ))
  return tf_example

def main(_):
  writer = tf.python_io.TFRecordWriter(FLAGS.output_path)

  # Read in your dataset to a variable
  #all_data_and_label_info = open(INPUT_YAML, 'rb')
  all_data_and_label_info = yaml.load(open(INPUT_YAML, 'rb').read())
  len_examples = len(all_data_and_label_info)
  print("Loaded ", len(all_data_and_label_info), "examples")

  for data_and_label_info in all_data_and_label_info:
    tf_example = create_tf_example(data_and_label_info)
    writer.write(tf_example.SerializeToString())

  writer.close()

if __name__ == '__main__':
  tf.app.run()
