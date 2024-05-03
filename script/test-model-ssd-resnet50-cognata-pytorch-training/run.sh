#!/bin/bash

echo "======================================================="

export PYTHONPATH=${CM_ML_MODEL_CODE_WITH_PATH}:${PYTHONPATH}
      
torchrun --nproc_per_node=${CM_ABTF_ML_MODEL_TRAINING_NPROC_PER_NODE} \
   ${CM_TMP_CURRENT_SCRIPT_PATH}/src/train.py \
   --model ${CM_ABTF_ML_MODEL_NAME} \
   --batch-size ${CM_ABTF_ML_MODEL_TRAINING_BATCH_SIZE} \
   --dataset ${CM_ABTF_DATASET} \
   --data-path ${CM_DATASET_MLCOMMONS_COGNATA_PATH} \
   --save-folder ${CM_ABTF_ML_MODEL_TRAINING_STORAGE} \
   --config ${CM_ABTF_ML_MODEL_CONFIG} \
   --save-name ${CM_ABTF_ML_MODEL_CONFIG} \
   --epoch ${CM_ABTF_ML_MODEL_EPOCH}

test $? -eq 0 || exit $?
