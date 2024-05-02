chmod 777 cognata
chmod 777 trained_models

docker run --rm -it -v $PWD/cognata:/cognata -v $PWD/trained_models:/trained_models --ipc=host --network=host    abtf-demo-cpu
