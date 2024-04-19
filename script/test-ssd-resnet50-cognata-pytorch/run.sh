#!/bin/bash

echo "======================================================="

${CM_PYTHON_BIN_WITH_PATH} ${CM_ML_MODEL_CODE_WITH_PATH}/test_image.py --pretrained-model "${CM_ML_MODEL_FILE_WITH_PATH}" --dataset ${CM_ABTF_DATASET} --config ${CM_ABTF_ML_MODEL_CONFIG} --input ${CM_INPUT_IMAGE} --output ${CM_OUTPUT_IMAGE} ${CM_ABTF_EXTRA_CMD}
test $? -eq 0 || exit $?
