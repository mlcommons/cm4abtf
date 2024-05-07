@echo off

echo =======================================================

set PYTHONPATH=%CM_ML_MODEL_CODE_WITH_PATH%;%PYTHONPATH%

%CM_PYTHON_BIN_WITH_PATH% %CM_TMP_CURRENT_SCRIPT_PATH%\src\test_dataset.py ^
  --pretrained-model "%CM_ML_MODEL_FILE_WITH_PATH%" ^
  --dataset %CM_ABTF_DATASET% ^
  --data-path %CM_DATASET_MLCOMMONS_COGNATA_PATH% ^
  --config %CM_ABTF_ML_MODEL_CONFIG% %CM_ABTF_EXTRA_CMD%
IF %ERRORLEVEL% NEQ 0 EXIT %ERRORLEVEL%
