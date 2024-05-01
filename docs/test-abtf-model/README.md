# Collective Mind interface and automation for ABTF

This repository contains [MLCommons CM](https://github.com/mlcommons/ck) automation recipes 
to make it easier to prepare and benchmark different versions of ABTF models 
(public or private) with MLPerf loadgen across different software and hardware.


## Install MLCommons CM automation framework

Follow [this online guide](https://access.cknowledge.org/playground/?action=install) 
to install CM for your OS with minimal dependencies.



## Install virtual environment

We tested these automation on Ubuntu and Windows.

We suggest you to create a virtual environment to avoid messing up your Python installation.

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



## Install required CM automation recipes

### CM repositories with automations

```bash
cm pull repo mlcommons@cm4mlops --checkout=dev
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

### Turn off debug info when running CM scripts

```bash
cm set cfg --key.script.silent
```

### Download ABTF models and sample image

```bash
cmr "download file _wget" --url="https://www.dropbox.com/scl/fi/cwyi6ukwih5qjblgh2m7x/baseline_8MP_ss_scales_fm1_5x5_all_ep60.pth?rlkey=okaq1s32leqnbjzloru206t50&st=70yoyy89&dl=0" --verify_ssl=no --md5sum=26845c3b9573ce115ef29dca4ae5be14
cmr "download file _wget" --url="https://www.dropbox.com/scl/fi/ljdnodr4buiqqwo4rgetu/baseline_4MP_ss_all_ep60.pth?rlkey=zukpgfjsxcjvf4obl64e72rf3&st=umfnx8go&dl=0" --verify_ssl=no --md5sum=75e56779443f07c25501b8e43b1b094f
cmr "download file _wget" --url="https://www.dropbox.com/scl/fi/9un2i2169rgebui4xklnm/baseline_8MP_ss_scales_all_ep60.pth?rlkey=sez3dnjep4waa09s5uy4r3wmk&st=z859czgk&dl=0" --verify_ssl=no --md5sum=1ab66f523715f9564603626e94e59c8c
cmr "download file _wget" --url="https://www.dropbox.com/scl/fi/0n7rmxxwqvg04sxk7bbum/0000008766.png?rlkey=mhmr3ztrlsqk8oa67qtxoowuh&dl=0" --verify_ssl=no --md5sum=903306a7c8bfbe6c1ca68fad6e34fe52 -s
```

### Download ABTF model code and register in CM cache

```bash
cmr "get ml-model abtf-ssd-pytorch"
```

or specific branch:

```bash
cmr "get ml-model abtf-ssd-pytorch" --model_code_git_branch=cognata
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


## Test ABTF Model with a sample image and prepare for loadgen


```bash
cmr "test abtf ssd-resnet50 cognata pytorch inference" --model=baseline_8MP_ss_scales_all_ep60.pth --config=baseline_8MP_ss_scales_all --input=0000008766.png --output=0000008766_prediction_test_8MP.jpg --num-classes=15
cmr "test abtf ssd-resnet50 cognata pytorch inference" --model=baseline_4MP_ss_all_ep60.pth --config=baseline_4MP_ss_all --input=0000008766.png --output=0000008766_prediction_test_4MP.jpg --num-classes=15
```

CM will load a workflow described by [this simple YAML]( https://github.com/mlcommons/cm4abtf/blob/dev/script/test-model-ssd-resnet50-cognata-pytorch-inference/_cm.yaml ),
call other CM scripts to detect, download or build missing deps for a given platform, prepare all environment variables and run benchmark
using the following [Python code](https://github.com/mlcommons/cm4abtf/blob/dev/script/test-model-ssd-resnet50-cognata-pytorch-inference/src/run_model.py).

You can run it in silent mode to skip CM workflow information using `-s` or `--silent` flag:
```bash
cmr "test abtf ssd-resnet50 cognata pytorch inference" --model=baseline_4MP_ss_all_ep60.pth --config=baseline_4MP_ss_all --input=0000008766.png --output=0000008766_prediction_test_4MP.jpg --num-classes=15 -s
```

You can dump internal CM state with resolved dependencies, their versions and reproducibility README by adding flag `--repro`.
You will find dump in the `cm-repro` directory of your current directory after script execution:

```
cmr "test abtf ssd-resnet50 cognata pytorch inference" --model=baseline_4MP_ss_all_ep60.pth --config=baseline_4MP_ss_all --input=0000008766.png --output=0000008766_prediction_test_4MP.jpg --num-classes=15 -s --repro
```



## Test ABTF model inference with loadgen


### Build MLPerf loadgen

```bash
cmr "get mlperf inference loadgen _copy" --version=main
```

### Run ABTF PyTorch model with loadgen

This example uses our [universal python loadgen harness with PyTorch and ONNX backends](https://github.com/mlcommons/cm4mlops/tree/main/script/app-loadgen-generic-python) with 1 real input saved as pickle:

```bash
cmr "test abtf ssd-resnet50 cognata pytorch inference" --model=baseline_8MP_ss_scales_all_ep60.pth --config=baseline_8MP_ss_scales_all --input=0000008766.png --output=0000008766_prediction_test_8MP.jpg --num-classes=15
cmr "generic loadgen python _pytorch _custom _cmc" --samples=5 --modelsamplepath=0000008766.png.cpu.pickle --modelpath=baseline_8MP_ss_scales_all_ep60.pth --modelcfg.num_classes=15 --modelcfg.config=baseline_8MP_ss_scales_all
```

```bash
cmr "test abtf ssd-resnet50 cognata pytorch inference" --model=baseline_4MP_ss_all_ep60.pth --config=baseline_4MP_ss_all --input=0000008766.png --output=0000008766_prediction_test_4MP.jpg --num-classes=15
cmr "generic loadgen python _pytorch _custom _cmc" --samples=5 --modelsamplepath=0000008766.png.cpu.pickle --modelpath=baseline_4MP_ss_all_ep60.pth --modelcfg.num_classes=15 --modelcfg.config=baseline_4MP_ss_all
```



## Export PyTorch ABTF model to ONNX

```bash
cmr "test abtf ssd-resnet50 cognata pytorch inference" --model=baseline_8MP_ss_scales_all_ep60.pth --config=baseline_8MP_ss_scales_all --input=0000008766.png --output=0000008766_prediction_test.jpg -s --export_model_to_onnx=baseline_8MP_ss_scales_all_ep60_opset17.onnx --export_model_to_onnx_opset=17
cmr "test abtf ssd-resnet50 cognata pytorch inference" --model=baseline_4MP_ss_all_ep60.pth --config=baseline_4MP_ss_all --input=0000008766.png --output=0000008766_prediction_test.jpg -s --export_model_to_onnx=baseline_4MP_ss_all_ep60_opset17.onnx --export_model_to_onnx_opset=17
```


## Run ABTF ONNX model with loadgen and random input


```bash
cm run script "generic loadgen python _onnxruntime" --samples=5 --modelpath=baseline_8MP_ss_scales_all_ep60_opset17.onnx --output_dir=results --repro -s
cm run script "generic loadgen python _onnxruntime" --samples=5 --modelpath=baseline_4MP_ss_all_ep60_opset17.onnx --output_dir=results --repro -s

```


## Try quantization via HuggingFace quanto package

We added simple example to do basic and automatic quantization of the model to int8 using [HugginFace's quanto package](https://github.com/huggingface/quanto):

```bash
cm run script "test abtf ssd-resnet50 cognata pytorch inference" 
     --ad.ml-model.model_code_git_branch=cognata \
     --model=baseline_8MP_ss_scales_all_ep60_state.pth \
     --config=baseline_8MP_ss_scales_all \
     --input=0000008766.png \
     --output=0000008766_prediction_test_8MP_quantized.jpg \
     --quantize_with_huggingface_quanto \
     --repro
```

## Run inference with quantized model

```
cm run script "test abtf ssd-resnet50 cognata pytorch inference" \
     --ad.ml-model.model_code_git_branch=cognata \
     --model=baseline_8MP_ss_scales_all_ep60_state_hf_quanto_qint8.pth \
     --config=baseline_8MP_ss_scales_all \
     --model_quanto \
     --input=0000008766.png \
     --output=0000008766_prediction_test_8MP_quantized_inference.jpg \
     --repro -s
```



## Archive of exported models

You can find all exported models in this [DropBox folder](https://www.dropbox.com/scl/fo/7ol30eled3vok3wwmt6h2/AJ7gbyZlLWjshENVzqTjZhI?rlkey=dafx4o5zd8l0395hzgxec5hcq&st=v5jrqr0u&dl=0).

You can download individual files via CM as follows:
```bash
cmr "download file _wget" --url="https://www.dropbox.com/scl/fi/yj3m4vpudmlqgkdkaatk0/baseline_4MP_ss_all_ep60_opset17.onnx?rlkey=r9i3ew2hfajzyssvmwb0vytbg&st=vkb3fwc7&dl=0" --verify_ssl=no --md5sum=7fbd70f7e1a0c0fbc1464f85351bce4b
```



## Test ABTF model with a Cognata sub-set

We have developed CM script to automate management and download of the MLCommons Cognata dataset. 
You just need to register [here](https://mlcommons.org/datasets/cognata/) and obtain a private URL that you will need to enter once:

```bash
cmr "get raw dataset mlcommons-cognata" --serial_numbers=10002_Urban_Clear_Morning --group_names=Cognata_Camera_01_8M --file_names=Cognata_Camera_01_8M_ann.zip;Cognata_Camera_01_8M_ann_laneline.zip;Cognata_Camera_01_8M.zip

cmr "test abtf ssd-resnet50 cognata pytorch inference _dataset" --model=baseline_4MP_ss_all_ep60.pth --config=baseline_4MP_ss_all

cmr "test abtf ssd-resnet50 cognata pytorch inference _dataset" --model=baseline_4MP_ss_all_ep60.pth --config=baseline_4MP_ss_all --visualize
```





## Benchmark accuracy of ABTF model with MLPerf loadgen


### Download Cognata data set via CM

We have developed a [CM automation recipe](https://github.com/mlcommons/cm4abtf/blob/dev/script/get-dataset-cognata-mlcommons) 
to download different sub-sets of the MLCommons Cognata data set - you just need to provide a private URL after registering to access 
this dataset [here](https://mlcommons.org/datasets/cognata/):


```bash
cmr "get raw dataset mlcommons-cognata" --serial_numbers=10002_Urban_Clear_Morning --group_names=Cognata_Camera_01_8M --file_names=Cognata_Camera_01_8M_ann.zip;Cognata_Camera_01_8M_ann_laneline.zip;Cognata_Camera_01_8M.zip```
```





## Roadmap

Current CM for ABTF roadmap [here](https://github.com/mlcommons/cm4abtf/issues/6).


## Feedback

Join MLCommons discord server
