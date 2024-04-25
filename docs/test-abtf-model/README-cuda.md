[ [Back to the main page](README.md) ]


## Prepare workflow to benchmark ABTF model on CUDA-based device

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
cmr "get cuda _toolkit _cudnn"
cmr "get cuda-devices"
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



## Test Model with a test image

```bash
cmr "test abtf ssd-resnet50 cognata pytorch inference _cuda" --input=0000008766.png --output=0000008766_prediction_test.jpg --config=baseline_8MP_ss_scales --num-classes=15
```

## Benchmark model with MLPerf loadgen

```bash
cmr "generic loadgen python _pytorch _cuda _custom _cmc" --samples=5 --modelsamplepath=0000008766.png.cuda.pickle --modelpath=baseline_8mp_ss_scales_ep15.pth --modelcfg.num_classes=13 --modelcfg.config=baseline_8MP_ss_scales
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
cmr "generic loadgen python _pytorch _cuda _custom _cmc" --samples=5 --modelsamplepath=0000008766.png.cpu.pickle \
  --modelpath=baseline_8mp_ss_scales_ep15.pth \
  --modelcfg.num_classes=13 \
  --modelcodepath="my-model-code" \
  --modelcfg.config=baseline_8MP_ss_scales
```





## Feedback

Join MLCommons discord or get in touch with developer: gfursin@cknowledge.org

