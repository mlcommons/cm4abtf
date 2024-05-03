[ [Back to the main page](README.md) ]

# Automating ABTF model training using CM and Docker

This repository contains [MLCommons CM](https://github.com/mlcommons/ck) automation recipes 
to make it easier to prepare and benchmark different versions of ABTF models 
(public or private) with MLPerf loadgen across different software and hardware.


## Start pre-generated CM-CUDA container for ABTF



## Import or download MLCommons Cognata data set

Import Cognata data set downloaded manually or via CM on the host and exposed to Docker via `/cognata` path:

```bash
cmr "get raw dataset mlcommons-cognata" --import=/cognata
```

or download Cognata data set:

```bash
cmr "get raw dataset mlcommons-cognata" --path=/cognata
```

## Run ABTF training


Using full dataset:

```bash
cmr "test abtf ssd-resnet50 cognata pytorch training _cuda" --config=baseline_4MP_ss_all 
```


Example with smaller dataset (on Windows with gloo):

```bash
cmr "test abtf ssd-resnet50 cognata pytorch training _cuda" --config=baseline_4MP_ss_all --torch_distributed_type=gloo --torch_distributed_init="" --dataset_folders=10002_Urban_Clear_Morning --dataset_cameras=Cognata_Camera_01_8M
```
