#!/bin/bash

echo "======================================================="

export PYTHONPATH=${CM_ML_MODEL_CODE_WITH_PATH}:${PYTHONPATH}
      
torchrun --nproc_per_node=1 \
   ${CM_TMP_CURRENT_SCRIPT_PATH}/src/evaluate.py \
   --model ${CM_ABTF_ML_MODEL_NAME} \
   --dataset ${CM_ABTF_DATASET} \
   --data-path ${CM_DATASET_MLCOMMONS_COGNATA_PATH} \
   --pretrained-model ${CM_ABTF_ML_MODEL_TRAINING_STORAGE} \
   --config ${CM_ABTF_ML_MODEL_CONFIG} \

test $? -eq 0 || exit $?
