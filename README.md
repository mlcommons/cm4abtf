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
. ABTF/bin/activate ; export CM_REPOS=$PWD/ABTF/CM
```
### Windows

```bash
python -m venv ABTF
call ABTF\Scripts\activate.bat & set CM_REPOS=%CD%\ABTF\CM
```

## Install CM automations 

### CM for MLOps

```bash
cm pull repo mlcommons@ck --checkout=dev
```

### CM for ABTF

```bash
cm pull repo cknowledge@cm4abtf --checkout=dev
```


## Download test ABTF model and image

```bash
cmr "download file _wget" --url="https://www.dropbox.com/scl/fi/od48qvnbqyfuy1z3aas84/baseline_8mp_ss_scales_ep15.pth?rlkey=d6ybe7g09g21pondmbd3pivzk&dl=0" --verify=no --md5sum=c36cb56b5f6bf8edbe64f9914506e09d
cmr "download file _wget" --url="https://www.dropbox.com/scl/fi/0n7rmxxwqvg04sxk7bbum/0000008766.png?rlkey=mhmr3ztrlsqk8oa67qtxoowuh&dl=0" --verify=no --md5sum=903306a7c8bfbe6c1ca68fad6e34fe52
```

### Install ABTF model PyTorch code 

```bash
cmr "get ml-model abtf-ssd-pytorch _local.baseline_8mp_ss_scales_ep15.pth"
```


Check CM cache:
```bash
cm show cache
```


## Prepare workflow to benchmark ABTF model on host CPU

Next commands prepare environment to benchmark host CPU.
Check these docs to benchmark other devices:
* [CUDA-based device](README-cuda.md)

### Build MLPerf loadgen

```bash
cmr "get mlperf inference loadgen _copy" --version=main
```

### Install or detect PyTorch

```bash
cmr "get generic-python-lib _torch"
cmr "get generic-python-lib _torchvision"
```



## Test Model with a test image

```bash
cmr "test abtf ssd-resnet50 cognata pytorch" --input=0000008766.png --output=0000008766_prediction_test.jpg --config=baseline_8MP_ss_scales --num-classes=13
```

## Benchmark model with MLPerf loadgen

```bash
cmr "generic loadgen python _pytorch _custom _cmc" --samples=5 --modelsamplepath=0000008766.png.cpu.pickle --modelpath=baseline_8mp.pth --modelcfg.num_classes=13 --modelcfg.config=baseline_8MP_ss_scales
```


## Benchmarking other models

Other ways to download public or private model code and weights:
```bash
cmr "get ml-model abtf-ssd-pytorch _skip_weights" --adr.abtf-ml-model-code-git-repo.env.CM_ABTF_MODEL_CODE_GIT_URL=https://github.com/mlcommons/abtf-ssd-pytorch
cmr "get ml-model abtf-ssd-pytorch _skip_weights" --model_code_git_url=https://github.com/mlcommons/abtf-ssd-pytorch --model_code_git_branch=cognata-cm
cmr "get ml-model abtf-ssd-pytorch _skip_weights _skip_code"
```

Other ways to run local (private) model:

```
cmr "generic loadgen python _pytorch _custom _cmc" --samples=5 --modelsamplepath=0000008766.png.cpu.pickle ^
  --modelpath=baseline_8mp.pth ^
  --modelcfg.num_classes=13 ^
  --modelcodepath="my-model-code" ^
  --modelcfg.config=baseline_8MP_ss_scales
```





## Feedback

Join MLCommons discord or get in touch with developer: gfursin@cknowledge.org

