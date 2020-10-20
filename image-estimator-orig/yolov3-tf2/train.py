#
#Copyright 2018 The Kubeflow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import numpy as np
import tensorflow as tf
from absl import app, flags, logging
from absl.flags import FLAGS
import numpy as np
import cv2
import os
import json

#from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import (ReduceLROnPlateau,
EarlyStopping,
ModelCheckpoint,
TensorBoard
)
from yolov3_tf2.models import (
YoloV3, YoloV3Tiny, YoloLoss,
yolo_anchors, yolo_anchor_masks,
yolo_tiny_anchors, yolo_tiny_anchor_masks
)
from yolov3_tf2.utils import freeze_all
import yolov3_tf2.dataset as dataset

flags.DEFINE_string('dataset', './data/voc2012_train.tfrecord', 'path to dataset')
flags.DEFINE_string('val_dataset', './data/voc2012_val.tfrecord', 'path to validation dataset')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_string('weights', '/mnt/checkpoints/yolov3_new.tf','path to weights file')
flags.DEFINE_string('classes', './data/voc.names', 'path to classes file')
flags.DEFINE_enum('mode', 'fit', ['fit', 'eager_fit', 'eager_tf'],
                                  'fit: model.fit, '
                                  'eager_fit: model.fit(run_eagerly=True), '
                                  'eager_tf: custom GradientTape')
flags.DEFINE_enum('transfer', 'none',
                                  ['none', 'darknet', 'no_output', 'frozen', 'fine_tune'],
                                  'none: Training from scratch, '
                                  'darknet: Transfer darknet, '
                                  'no_output: Transfer all but output, '
                                  'frozen: Transfer and freeze all, '
                                  'fine_tune: Transfer all and freeze darknet only')
flags.DEFINE_integer('size', 416, 'image size')
flags.DEFINE_integer('epochs', 3, 'number of epochs')
flags.DEFINE_integer('batch_size', 32, 'batch size')
flags.DEFINE_float('learning_rate', 1e-3, 'learning rate')
flags.DEFINE_integer('num_classes', 20, 'number of classes in the model')
flags.DEFINE_integer('weights_num_classes', None, 'specify num class for `weights` file if different, '
                                         'useful in transfer learning with different number of classes')
flags.DEFINE_string('model_dir', '/mnt/estim_trained_model11/.', 'path to saved model')
                                         
#tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

def input_fn():

       #BUFFER_SIZE = 10000
       
        anchors = yolo_anchors
        anchor_masks = yolo_anchor_masks
        
        if FLAGS.dataset:
                   train_dataset = dataset.load_tfrecord_dataset(
                                FLAGS.dataset, FLAGS.classes, FLAGS.size)
        else:
                train_dataset = dataset.load_fake_dataset()
        train_dataset = train_dataset.shuffle(buffer_size=1000)
        train_dataset = train_dataset.repeat(FLAGS.epochs)
        train_dataset = train_dataset.batch(FLAGS.batch_size)
        
        return train_dataset.map(lambda x, y: (dataset.transform_images(x, FLAGS.size),
            dataset.transform_targets(y, anchors, anchor_masks, FLAGS.size)))

def main(args):
  print('Using %s to store checkpoints.' %FLAGS.model_dir)
  
  anchors = yolo_anchors
  anchor_masks = yolo_anchor_masks

  if FLAGS.tiny:
           model = YoloV3Tiny(FLAGS.size, training=True,
                           classes=FLAGS.num_classes)
                 #anchors = yolo_tiny_anchors
                 #anchor_masks = yolo_tiny_anchor_masks
  else:
           model = YoloV3(FLAGS.size, training=True, classes=FLAGS.num_classes)
                
                
  optimizer = tf.keras.optimizers.Adam(lr=FLAGS.learning_rate)
  loss = [YoloLoss(anchors[mask], classes=FLAGS.num_classes)
                                        for mask in anchor_masks]
                                        
  model.summary()
                                        
  model.compile(optimizer=optimizer, loss=loss,
                             run_eagerly=(FLAGS.mode == 'eager_fit'))
  #tf.keras.backend.set_learning_phase(True)
                             
                                     
  # Define DistributionStrategies and convert the Keras Model to an
  # Estimator that utilizes these DistributionStrateges.
  # Evaluator is a single worker, so using MirroredStrategy.
  
  config = tf.estimator.RunConfig(
         train_distribute=tf.distribute.experimental.ParameterServerStrategy(),
         eval_distribute=tf.distribute.MirroredStrategy())
  
  keras_estimator = tf.keras.estimator.model_to_estimator(
      keras_model=model, config=config, model_dir=FLAGS.model_dir)

  # Train and evaluate the model. Evaluation will be skipped if there is not an
  # "evaluator" job in the cluster.
  tf.estimator.train_and_evaluate(
      keras_estimator,
      train_spec=tf.estimator.TrainSpec(input_fn=input_fn),
      eval_spec=tf.estimator.EvalSpec(input_fn=input_fn))


if __name__ == '__main__':
  #tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
  import time
  start_time = time.time()
  tf.compat.v1.app.run(main)
  end_time = time.time() - start_time
  print("Completed at %s"%end_time)
