@echo off

echo =======================================================

set PYTHONPATH=%CM_ML_MODEL_CODE_WITH_PATH%;%PYTHONPATH%

%CM_PYTHON_BIN_WITH_PATH% %CM_TMP_CURRENT_SCRIPT_PATH%\src\run_model.py ^
  --pretrained-model "%CM_ML_MODEL_FILE_WITH_PATH%" ^
  --dataset %CM_ABTF_DATASET% ^
  --config %CM_ABTF_ML_MODEL_CONFIG% %CM_ABTF_EXTRA_CMD%
IF %ERRORLEVEL% NEQ 0 EXIT %ERRORLEVEL%
