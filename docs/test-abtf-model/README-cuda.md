[ [Back to the main page](README.md) ]


## Benchmark ABTF model on CUDA-based device using CM without Docker

### Prerequisites

* We expect that you already have CUDA driver installed
* Tested with 
  * Python 3.11.x (3.12+ currently doesn't work)
  * CUDA 11.8 and 12.1 (with cuDNN)
  * torch 2.2.2 and 2.3.0
  * torchvision 0.17.1 and 0.18.0 
* Didn't work on Windows with
  * Python 3.12.x
  * CUDA 12.1 with cuDNN
  * torch 2.3.0
  * torchvision 0.18.0 


### Detect or install CUDA toolkit and libraries


```bash
cmr "get cuda _toolkit"
cmr "get cuda-devices"
```

If you need to use cuDNN, use the following command:

```bash
cmr "get cuda _toolkit _cudnn"
```

If cuDNN is not installed, you can download it from the website and register via CM as follows:
```bash
cmr "get cudnn" --tar_file={FULL PATH TO cudnn TAR file}
```

### Build MLPerf loadgen

```bash
cmr "get mlperf inference loadgen _copy" --version=main
```


### Install or detect PyTorch and PyTorchVision

*TBD: better automation of CUDA version detection and passing extra-index-url*

#### CUDA 11.8

```bash
cmr "get generic-python-lib _torch_cuda" --extra-index-url=https://download.pytorch.org/whl/cu118 --force-install
cmr "get generic-python-lib _torchvision_cuda" --extra-index-url=https://download.pytorch.org/whl/cu118 --force-install
```

#### CUDA 12.1

```bash
cmr "get generic-python-lib _torch_cuda" --extra-index-url=https://download.pytorch.org/whl/cu121 --force-install --version=2.2.2
cmr "get generic-python-lib _torchvision_cuda" --extra-index-url=https://download.pytorch.org/whl/cu121 --force-install --version=0.17.1
```

#### CUDA 12.2

```bash
cmr "get generic-python-lib _torch_cuda" --extra-index-url=https://download.pytorch.org/whl/cu122 --force-install --version=2.2.2
cmr "get generic-python-lib _torchvision_cuda" --extra-index-url=https://download.pytorch.org/whl/cu122 --force-install --version=0.17.1
```


## Test Model with a test image

```bash
cmr "test abtf ssd-resnet50 cognata pytorch inference _cuda" \
     --model=baseline_4MP_ss_all_ep60.pth \
     --config=baseline_4MP_ss_all \
     --input=0000008766.png 
     --output=0000008766_prediction_test.jpg \
     --repro -s
```

## Benchmark model with MLPerf loadgen

```bash
cm run script "generic loadgen python _pytorch _cuda _custom _cmc" \
     --samples=5 \
     --modelsamplepath=0000008766.png.cuda.pickle \
     --modelpath=baseline_4MP_ss_all_ep60.pth \
     --modelcfg.num_classes=15 \
     --modelcfg.config=baseline_4MP_ss_all \
     --output_dir=results \
     --repro -s
```


## Check with Cognata demo

```bash
cmr "get raw dataset mlcommons-cognata" --serial_numbers=10002_Urban_Clear_Morning --group_names=Cognata_Camera_01_8M --file_names=Cognata_Camera_01_8M_ann.zip;Cognata_Camera_01_8M_ann_laneline.zip;Cognata_Camera_01_8M.zip
cmr "test abtf ssd-resnet50 cognata pytorch inference _cuda _dataset" --model=baseline_4MP_ss_all_ep60.pth --config=baseline_4MP_ss_all --visualize
```


## Feedback

Join MLCommons discord or get in touch with developers: Radoyeh Shojaei and Grigori Fursin

