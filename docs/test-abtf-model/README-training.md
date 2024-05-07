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

## Run ABTF model training


Using full dataset:

```bash
cmr "demo abtf ssd-resnet50 cognata pytorch training _cuda" --config=baseline_4MP_ss_all 
```

or partial dataset:
```bash
cmr "demo abtf ssd-resnet50 cognata pytorch training _cuda" --config=baseline_4MP_ss_all --dataset_folders=10002_Urban_Clear_Morning --dataset_cameras=Cognata_Camera_01_8M
```

or on Windows with gloo:

```bash
cmr "demo abtf ssd-resnet50 cognata pytorch training _cuda" --config=baseline_4MP_ss_all --torch_distributed_type=gloo --torch_distributed_init="" --dataset_folders=10002_Urban_Clear_Morning --dataset_cameras=Cognata_Camera_01_8M
```


See related [CM script](https://github.com/mlcommons/cm4abtf/tree/dev/script/demo-ml-model-abtf-cognata-pytorch-training).



## Run ABTF model evaluation

```bash
cmr "download file _wget" --url="https://www.dropbox.com/scl/fi/ljdnodr4buiqqwo4rgetu/baseline_4MP_ss_all_ep60.pth?rlkey=zukpgfjsxcjvf4obl64e72rf3&st=umfnx8go&dl=0" --verify_ssl=no --md5sum=75e56779443f07c25501b8e43b1b094f
cmr "demo abtf ssd-resnet50 cognata pytorch evaluation _cuda" --config=baseline_4MP_ss_all --pretrained_model=baseline_4MP_ss_all_ep60.pth
```

Run on partial dataset (just for a test):
```bash
cmr "demo abtf ssd-resnet50 cognata pytorch evaluation _cuda" --config=baseline_4MP_ss_all --dataset_folders=10002_Urban_Clear_Morning --dataset_cameras=Cognata_Camera_01_8M --pretrained_model=baseline_4MP_ss_all_ep60.pth --force_cognata_labels=yes
```
or on Windows:
```
cmr "demo abtf ssd-resnet50 cognata pytorch evaluation _cuda" --config=baseline_4MP_ss_all --torch_distributed_type=gloo --torch_distributed_init="" --dataset_folders=10002_Urban_Clear_Morning --dataset_cameras=Cognata_Camera_01_8M --pretrained_model=baseline_4MP_ss_all_ep60.pth --force_cognata_labels=yes
```


See related [CM script](https://github.com/mlcommons/cm4abtf/tree/dev/script/demo-ml-model-abtf-cognata-pytorch-evaluation).

## Run ABTF model test

```bash
cmr "download file _wget" --url="https://www.dropbox.com/scl/fi/ljdnodr4buiqqwo4rgetu/baseline_4MP_ss_all_ep60.pth?rlkey=zukpgfjsxcjvf4obl64e72rf3&st=umfnx8go&dl=0" --verify_ssl=no --md5sum=75e56779443f07c25501b8e43b1b094f
cmr "demo abtf ssd-resnet50 cognata pytorch test _cuda" --config=baseline_4MP_ss_all --model=baseline_4MP_ss_all_ep60.pth
```

Run on partial dataset (just for a test):
```bash
cmr "demo abtf ssd-resnet50 cognata pytorch test _cuda" --config=baseline_4MP_ss_all --dataset_folders=10002_Urban_Clear_Morning --dataset_cameras=Cognata_Camera_01_8M --pretrained_model=baseline_4MP_ss_all_ep60.pth --force_cognata_labels=yes
```
or on Windows:
```
cmr "demo abtf ssd-resnet50 cognata pytorch test _cuda" --config=baseline_4MP_ss_all --dataset_folders=10002_Urban_Clear_Morning --dataset_cameras=Cognata_Camera_01_8M --pretrained_model=baseline_4MP_ss_all_ep60.pth --force_cognata_labels=yes
```

See related [CM script](https://github.com/mlcommons/cm4abtf/tree/dev/script/demo-ml-model-abtf-cognata-pytorch-test).
