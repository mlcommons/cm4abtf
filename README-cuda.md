[ [Back to the main page](README.md) ]

*Tested with PyTorch 2.2.2 and CUDA 11.8 and 12.1*

## Prepare workflow to benchmark ABTF model on CUDA-based device

We expect that you already have CUDA driver installed

### Detect or install CUDA toolkit and libraries

``bash
cmr "get cuda _toolkit _cudnn"
cmr "get cuda-devices"
```

### Install or detect PyTorch and PyTorchVision

#### CUDA 11.8

```bash
cmr "get generic-python-lib _torch_cuda" --extra-index-url=https://download.pytorch.org/whl/cu118 --force-install
cmr "get generic-python-lib _torchvision_cuda" --extra-index-url=https://download.pytorch.org/whl/cu118 --force-install
```

#### CUDA 12.1

```bash
cmr "get generic-python-lib _torch_cuda" --extra-index-url=https://download.pytorch.org/whl/cu121 --force-install
cmr "get generic-python-lib _torchvision_cuda" --extra-index-url=https://download.pytorch.org/whl/cu121 --force-install
```





## Test Model with a test image

```bash
cmr "test abtf ssd-resnet50 cognata pytorch _cuda" --input=0000008766.png --output=0000008766_prediction_test.jpg --config=baseline_8MP_ss_scales --num-classes=13
```

## Benchmark model with MLPerf loadgen

```bash
cmr "generic loadgen python _pytorch _cuda _custom _cmc" --samples=5 --modelsamplepath=0000008766.png.cuda.pickle --modelpath=baseline_8mp.pth --modelcfg.num_classes=13 --modelcfg.config=baseline_8MP_ss_scales
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
cmr "generic loadgen python _pytorch _cuda _custom _cmc" --samples=5 --modelsamplepath=0000008766.png.cpu.pickle ^
  --modelpath=baseline_8mp.pth ^
  --modelcfg.num_classes=13 ^
  --modelcodepath="my-model-code" ^
  --modelcfg.config=baseline_8MP_ss_scales
```





## Feedback

Join MLCommons discord or get in touch with developer: gfursin@cknowledge.org

