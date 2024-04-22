# Collective Mind interface and automation for ABTF

This repository contains [MLCommons CM](https://github.com/mlcommons/ck) automation recipes 
to make it easier to prepare and benchmark different versions of ABTF models 
(public or private) with MLPerf loadgen across different software and hardware.


## Install MLCommons CM automation framework

Follow [this online guide](https://access.cknowledge.org/playground/?action=install) 
to install CM for your OS with minimal dependencies.



## Install virtual environment

We suggest to create a virtual environment to avoid messing up your Python installation.
All CM repositories, artifacts and cache will be resided inside this virtual environment
and can be removed at any time without influencing your own environment!

### Linux

```bash
python3 -m venv ABTF
mkdir ABTF/work
. ABTF/bin/activate ; export CM_REPOS=$PWD/ABTF/CM ; cd ABTF/work
```
### Windows

```bash
python -m venv ABTF
md ABTF\work
call ABTF\Scripts\activate.bat & set CM_REPOS=%CD%\ABTF\CM & cd ABTF\work
```



## Install CM automations 

### CM for MLOps and DevOps

```bash
cm pull repo mlcommons@cm4mlops --checkout=dev
```

### CM for ABTF

```bash
cm pull repo mlcommons@cm4abtf --checkout=dev
```

### Show installed CM repositories

```bash
cm show repo
```

### Find a specific repo

```bash
cm find repo mlcommons@cm4abtf
```


### Update all repositories at any time

```bash
cm pull repo
```


### Download ABTF models and sample image

```bash
cmr "download file _wget" --url="https://www.dropbox.com/scl/fi/9un2i2169rgebui4xklnm/baseline_8MP_ss_scales_all_ep60.pth?rlkey=sez3dnjep4waa09s5uy4r3wmk&st=z859czgk&dl=0" --verify=no --md5sum=1ab66f523715f9564603626e94e59c8c
cmr "download file _wget" --url="https://www.dropbox.com/scl/fi/od48qvnbqyfuy1z3aas84/baseline_8mp_ss_scales_ep15.pth?rlkey=d6ybe7g09g21pondmbd3pivzk&dl=0" --verify=no --md5sum=c36cb56b5f6bf8edbe64f9914506e09d -s
cmr "download file _wget" --url="https://www.dropbox.com/scl/fi/ljdnodr4buiqqwo4rgetu/baseline_4mp_ss_all_ep60.pth?rlkey=zukpgfjsxcjvf4obl64e72rf3&st=umfnx8go&dl=0" --verify=no --md5sum=75e56779443f07c25501b8e43b1b094f -s
cmr "download file _wget" --url="https://www.dropbox.com/scl/fi/0n7rmxxwqvg04sxk7bbum/0000008766.png?rlkey=mhmr3ztrlsqk8oa67qtxoowuh&dl=0" --verify=no --md5sum=903306a7c8bfbe6c1ca68fad6e34fe52 -s
```

### Download ABTF model code and register in CM cache

```bash
cmr "get ml-model abtf-ssd-pytorch"
```

### Check the state of the CM cache

```bash
cm show cache
```

### Find model code

```bash
cm show cache --tags=ml-model,abtf
```


## Prepare workflow to benchmark ABTF model on host CPU

Next commands prepare environment to benchmark host CPU.
Check these docs to benchmark other devices:
* [CUDA-based device](README-cuda.md)




### Detect python from virtual env

```bash
cmr "get python3" --quiet
```

### Install or detect PyTorch


```bash
cmr "get generic-python-lib _torch"
cmr "get generic-python-lib _torchvision"
```

If you want to install a specific version of PyTorch, you can specify it as follows:

```bash
cmr "get generic-python-lib _torch" --version=2.2.0
cmr "get generic-python-lib _torchvision" --version=0.17.0
```


## Run ABTF Model with a test image and prepare for loadgen

```bash
cmr "test abtf ssd-resnet50 cognata pytorch inference" --model=baseline_8MP_ss_scales_all_ep60.pth --config=baseline_8MP_ss_scales_all --input=0000008766.png --output=0000008766_prediction_test.jpg
cmr "test abtf ssd-resnet50 cognata pytorch inference" --model=baseline_8mp_ss_scales_ep15.pth --config=baseline_8MP_ss_scales --input=0000008766.png --output=0000008766_prediction_test.jpg --num-classes=13
```

CM will load a workflow described by [this simple YAML](https://github.com/mlcommons/cm4abtf/blob/dev/script/test-ssd-resnet50-cognata-pytorch/_cm.yaml),
call other CM scripts to detect or build missing deps for a given platform, prepare all environment variables and run benchmark.

You can run it in silent mode to skip CM workflow information using `-s` or `--silent` flag:
```bash
cmr "test abtf ssd-resnet50 cognata pytorch inference" --model=baseline_8MP_ss_scales_all_ep60.pth --config=baseline_8MP_ss_scales_all --input=0000008766.png --output=0000008766_prediction_test.jpg -s
```

## Benchmark performance of ABTF model with MLPerf loadgen

### Build MLPerf loadgen

```bash
cmr "get mlperf inference loadgen _copy" --version=main
```

### Run ABTF model with loadgen

```bash
cmr "test abtf ssd-resnet50 cognata pytorch inference" --model=baseline_8MP_ss_scales_all_ep60.pth --config=baseline_8MP_ss_scales_all --input=0000008766.png --output=0000008766_prediction_test.jpg
cmr "generic loadgen python _pytorch _custom _cmc" --samples=5 --modelsamplepath=0000008766.png.cpu.pickle --modelpath=baseline_8MP_ss_scales_all_ep60.pth --modelcfg.num_classes=15 --modelcfg.config=baseline_8MP_ss_scales_all
```

or older version

```
cmr "test abtf ssd-resnet50 cognata pytorch inference" --model=baseline_8mp_ss_scales_ep15.pth --config=baseline_8MP_ss_scales --input=0000008766.png --output=0000008766_prediction_test.jpg --num-classes=13
cmr "generic loadgen python _pytorch _custom _cmc" --samples=5 --modelsamplepath=0000008766.png.cpu.pickle --modelpath=baseline_8mp_ss_scales_ep15.pth --modelcfg.num_classes=13 --modelcfg.config=baseline_8MP_ss_scales
```


## Benchmarking other models

Other ways to download public or private model code and weights:
```bash
cmr "get ml-model abtf-ssd-pytorch _skip_weights" --adr.abtf-ml-model-code-git-repo.env.CM_ABTF_MODEL_CODE_GIT_URL=https://github.com/mlcommons/abtf-ssd-pytorch
cmr "get ml-model abtf-ssd-pytorch _skip_weights" --model_code_git_url=https://github.com/mlcommons/abtf-ssd-pytorch --model_code_git_branch=cognata-cm
cmr "get ml-model abtf-ssd-pytorch _skip_weights _skip_code"
```

Other ways to run local (private) model:

You can first copy ABTF model code from GitHub to your local directory `my-model-code`.

```
cmr "generic loadgen python _pytorch _custom _cmc" --samples=5 --modelsamplepath=0000008766.png.cpu.pickle \
  --modelpath=baseline_8mp_ss_scales_ep15.pth \
  --modelcfg.num_classes=13 \
  --modelcodepath="my-model-code" \
  --modelcfg.config=baseline_8MP_ss_scales
```


## Benchmark accuracy of ABTF model with MLPerf loadgen

### Download Cognata data set via CM

We have developed a [CM automation recipe](https://github.com/mlcommons/cm4abtf/blob/dev/script/get-dataset-cognata-mlcommons/customize.py) 
to download different sub-sets of the MLCommons Cognata data set - you just need to provide a private URL after registering to access 
this dataset [here](https://mlcommons.org/datasets/cognata/):


```bash
cmr "get raw dataset mlcommons-cognata" --serial_numbers=10002_Urban_Clear_Morning --group_names=Cognata_Camera_02_8M
```





## Feedback

Join MLCommons discord or get in touch with developer: gfursin@cknowledge.org

