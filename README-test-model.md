# CM automation for ABTF-MLPerf

*Testing ABTF SSD PyTorch model via the [MLCommons CM automation meta-framework](https://github.com/mlcommons/ck).*

## Install CM

Follow [this online guide](https://access.cknowledge.org/playground/?action=install) to install CM for your OS.

## Install virtual environment

We suggest to create a virtual environment to avoid messing up your Python installation:

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

## Install all CM automation recipes

Pull [main MLOps automation recipes](https://access.cknowledge.org/playground/?action=scripts) from MLCommons:

```bash
cm pull repo mlcommons@ck --checkout=dev
```

Pull this CM repository with automation recipes for the MLCommons-ABTF benchmark:

```bash
cm pull repo cknowledge@cm4abtf
```

## Clean CM cache

Clean CM cache if you want to start from scratch

```bash
cm rm cache -f
```









Download private test image `0000008766.png` and model `baseline_8mp.pth` to your local directory.


Import `baseline_8mp.pth` to CM:
```bash
cmr "get ml-model abtf-ssd-pytorch _local.baseline_8mp.pth"
```

Get Git repo with ABTF SSD-ResNet50 PyTorch model:

```bash
cmr "get git repo _repo.https://github.com/mlcommons/abtf-ssd-pytorch" --env.CM_GIT_BRANCH=cognata-cm --extra_cache_tags=abtf,ssd,pytorch,cm-model --env.CM_GIT_CHECKOUT_PATH_ENV_NAME=CM_ABTF_SSD_PYTORCH
```

Make test prediction:

```bash
cmr "test abtf ssd-resnet50 cognata pytorch" --input=0000008766.png --output=0000008766_prediction_test.jpg --config=baseline_8MP
```

Export PyTorch model to ONNX:
```bash
cmr "test abtf ssd-resnet50 cognata pytorch" --input=0000008766.png --output=0000008766_prediction_test.jpg --config=baseline_8MP --export_model=baseline_8mp.onnx
```

Test exported ONNX model with LoadGen (performance):
```bash
cm run script "python app loadgen-generic _onnxruntime" --modelpath=baseline_8mp.onnx --samples=1 --quiet
```


Test different versions of PyTorch
```bash
cmr "install python-venv" --name=abtf2
cmr "test abtf ssd-resnet50 cognata pytorch" --adr.python.name=abtf2 --adr.torch.version=1.13.1 --adr.torchvision.version=0.14.1 --input=0000008766.png --output=0000008766_prediction_test.jpg --config=baseline_8MP
```

## TBD

### Main features

* Test PyTorch model with Python LoadGen
* Test PyTorch model with [C++ loadgen](https://github.com/mlcommons/ck/tree/master/cm-mlops/script/app-mlperf-inference-mlcommons-cpp)
* Automate loading of Cognata dataset via CM
* Add Cognata dataset to loadgen
* Process PyTorch model with MLPerf inference infrastructure for SSD-ResNet50
* Add support for MLCommons Croissant

### Testing docker

```bash
cm docker script --tags=test,abtf,ssd-pytorch,_cognata --docker_cm_repo=ctuning@mlcommons-ck --env.CM_GH_TOKEN={TOKEN} --input=road.jpg --output=road_ssd.jpg
```

```bash
cm docker script --tags=test,abtf,ssd-pytorch,_cognata --docker_cm_repo=ctuning@mlcommons-ck --docker_os=ubuntu --docker_os_version=23.04 --input=road.jpg --output=road_ssd.jpg 
```
TBD: pass file to CM docker: [meta](https://github.com/mlcommons/ck/blob/master/cm-mlops/script/build-mlperf-inference-server-nvidia/_cm.yaml#L197).

## CM automation developers

* [Grigori Fursin](https://cKnowledge.org/gfursin) (MLCommons Task Force on Automation and Reproducibility)
