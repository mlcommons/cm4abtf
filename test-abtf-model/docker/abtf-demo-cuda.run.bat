docker run --gpus all --rm -it -v %CD%\cognata:/cognata -v %CD%\trained_models:/trained_models --ipc=host --network=host    abtf-demo-cuda
