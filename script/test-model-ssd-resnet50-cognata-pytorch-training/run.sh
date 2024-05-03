#!/bin/bash

echo "======================================================="

cd ${CM_ML_MODEL_CODE_WITH_PATH}
      
torchrun --nproc_per_node=1  train.py --model ssd --batch-size 1 --dataset Cognata --data-path ${CM_DATASET_MLCOMMONS_COGNATA_PATH} --save-folder trained_models --config test_8MP --save-name test_8mp --epoch 2
test $? -eq 0 || exit $?
