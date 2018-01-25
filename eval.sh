#!/bin/bash
set -e

# Paths to model and evaluation results
TRAIN_DIR=./topcoder_logs/1.25/run_X/
TEST_DIR=${TRAIN_DIR}/eval

# Where the dataset is saved to.
DATASET_DIR=/mnt/data/tensorflow/data
LABELS_FILE=
NUM_CLASSES=200
# Run evaluation (using slim.evaluation.evaluate_once)
CONTINUE=1

while [ "$CONTINUE" -ne 0 ]
do

python eval_image_classifier.py \
  --checkpoint_path=${TRAIN_DIR} \
  --eval_dir=${TEST_DIR} \
  --preprocessing_name=preprocess224 \
  --split_name=validation \
  --dataset_dir=${DATASET_DIR} \
  --file_pattern=topc_%s_*.tfrecord \
  --file_pattern=topc \
  --labels_file=${LABELS_FILE} \
  --model_name=mobilenet_v1 \
  --batch_size=64 \
  --n_classes=${NUM_CLASSES}

echo "sleeping for next run"
sleep 600
done