docker run --gpus all --rm -it -v $PWD/cognata:/cognata -v $PWD/trained_models:/trained_models --ipc=host --network=host    abtf-demo-cuda
