#!/bin/bash

set -x

cd /opt/yolov3-tf2/

#wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar -O ./data/voc2012_raw.tar
#wget http://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar -O ./data/voc2012_raw.tar
#cp ../VOCtest_06-Nov-2007.tar ./data/voc2012_raw.tar
#mkdir -p ./data/voc2012_raw
#tar -xvf ./data/voc2012_raw.tar -C ./data/voc2012_raw
#ls ./data/voc2012_raw/VOCdevkit/VOC2012

python3 tools/voc2012.py \
  --data_dir '/mnt/tfjob/VOCdevkit/VOC2012/' \
  --split train \
  --output_file ./data/voc2012_train.tfrecord

python3  tools/voc2012.py \
  --data_dir '/mnt/tfjob/VOCdevkit/VOC2012/' \
  --split val \
  --output_file ./data/voc2012_val.tfrecord

python3 convert.py
time python3 train.py

#cp -r trained_model/ /mnt/tfjob
#cp -r checkpoints/ /mnt/tfjob/checkpts
#cp -r logs/ /mnt/tfjob/logs1
