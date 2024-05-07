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
cmr "demo abtf ssd-resnet50 cognata pytorch inference _cuda" \
     --model=baseline_4MP_ss_all_ep60.pth \
     --config=baseline_4MP_ss_all \
     --input=0000008766.png 
     --output=0000008766_prediction_test.jpg \
     --repro -s
```

## Check with Cognata demo

```bash
cmr "get raw dataset mlcommons-cognata" --serial_numbers=10002_Urban_Clear_Morning --group_names=Cognata_Camera_01_8M --file_names="Cognata_Camera_01_8M_ann.zip;Cognata_Camera_01_8M_ann_laneline.zip;Cognata_Camera_01_8M.zip"
cmr "demo abtf ssd-resnet50 cognata pytorch inference _cuda _dataset" --model=baseline_4MP_ss_all_ep60.pth --config=baseline_4MP_ss_all --visualize
```



## Benchmark model with MLPerf loadgen (just 1 performance sample)

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




## Benchmark accuracy of ABTF model with MLPerf loadgen


### Download Cognata data set via CM

We have developed a [CM automation recipe](https://github.com/mlcommons/cm4abtf/blob/dev/script/get-dataset-cognata-mlcommons) 
to download different sub-sets of the MLCommons Cognata data set - you just need to provide a private URL after registering to access 
this dataset [here](https://mlcommons.org/datasets/cognata/):


```bash
cmr "get raw dataset mlcommons-cognata" --serial_numbers=10002_Urban_Clear_Morning --group_names=Cognata_Camera_01_8M --file_names="Cognata_Camera_01_8M_ann.zip;Cognata_Camera_01_8M_ann_laneline.zip;Cognata_Camera_01_8M.zip"
```
or in a more compact way via variation

```bash
cmr "get raw dataset mlcommons-cognata _abtf-demo"
```

and to some specific path
```bash
cmr "get raw dataset mlcommons-cognata _abtf-demo" --path=/cognata
```

or import the existing one
```bash
cmr "get raw dataset mlcommons-cognata _abtf-demo" --import=/cognata
```

### Run basic MLPerf harness for ABTF model

We've prototyped a basic MLPerf harness with CM automation to benchmark ABTF model with Cognata data set. 
See [CM script](https://github.com/mlcommons/cm4abtf/tree/dev/script/demo-ml-model-abtf-cognata-pytorch-loadgen) 
and [MLPerf Python harness for Cognata and ABTF model](https://github.com/mlcommons/cm4abtf/tree/dev/script/demo-ml-model-abtf-cognata-pytorch-loadgen/ref/python).
It should work on Linux and Windows with CPU and CUDA GPU.

You can run performance test with the ABTF model and Cognata datset on CPU as follows (use --count to select the number of samples):

```bash
cmr "download file _wget" --url="https://www.dropbox.com/scl/fi/9un2i2169rgebui4xklnm/baseline_8MP_ss_scales_all_ep60.pth?rlkey=sez3dnjep4waa09s5uy4r3wmk&st=z859czgk&dl=0" --verify_ssl=no --md5sum=1ab66f523715f9564603626e94e59c8c

cmr "get raw dataset mlcommons-cognata _abtf-demo" --import=/cognata

cm run script --tags=run-mlperf-inference,demo,abtf-model,_abtf-demo-model,_pytorch,_cuda \
   --dataset=cognata-8mp-pt \
   --model=$PWD/baseline_8MP_ss_scales_all_ep60.pth \
   --env.CM_ABTF_ML_MODEL_CONFIG=baseline_8MP_ss_scales_all \
   --env.CM_ABTF_NUM_CLASSES=15 \
   --env.CM_DATASET_MLCOMMONS_COGNATA_SERIAL_NUMBERS=10002_Urban_Clear_Morning \
   --env.CM_DATASET_MLCOMMONS_COGNATA_GROUP_NAMES=Cognata_Camera_01_8M \
   --env.CM_ABTF_ML_MODEL_TRAINING_FORCE_COGNATA_LABELS=yes \
   --env.CM_ABTF_ML_MODEL_SKIP_WARMUP=yes \
   --max_batchsize=1 \
   --count=2 \
   --precision=float32 \
   --implementation=mlcommons-python \
   --scenario=Offline \
   --mode=performance \
   --power=no \
   --adr.python.version_min=3.8 \
   --adr.compiler.tags=gcc \
   --output=$PWD/results \
   --rerun \
   --clean
```

You should see the following output:
```bash
======================================================================

Loading model ...


Loading dataset and preprocessing if needed ...
* Dataset path: /cognata
* Preprocessed cache path: /home/cmuser/CM/repos/mlcommons@cm4abtf/script/demo-ml-model-abtf-cognata-pytorch-loadgen/tmp-preprocessed-dataset


Cognata folders: ['10002_Urban_Clear_Morning']
Cognata cameras: ['Cognata_Camera_01_8M']


Scanning Cognata dataset ...
  Number of files found: 1020
  Time: 2.36 sec.

Preloading and preprocessing Cognata dataset on the fly ...
  Time: 0.62 sec.

TestScenario.Offline qps=3.73, mean=0.5024, time=0.537, queries=2, tiles=50.0:0.5024,80.0:0.5056,90.0:0.5067,95.0:0.5072,99.0:0.5076,99.9:0.5077
```

You can run MLPerf accuracy test by changing `--mode=performance` to `--mode=accuracy` 
in the above command line and rerun it. You should the output similar to the following:

```bash
======================================================================

Loading model ...


Loading dataset and preprocessing if needed ...
* Dataset path: /cognata
* Preprocessed cache path: /home/cmuser/CM/repos/mlcommons@cm4abtf/script/demo-ml-model-abtf-cognata-pytorch-loadgen/tmp-preprocessed-dataset


Cognata folders: ['10002_Urban_Clear_Morning']
Cognata cameras: ['Cognata_Camera_01_8M']


Scanning Cognata dataset ...
  Number of files found: 1020
  Time: 2.36 sec.

Preloading and preprocessing Cognata dataset on the fly ...
  Time: 0.64 sec.

=================================================
{   'classes': tensor([ 2,  3,  4,  5,  9, 10], dtype=torch.int32),
    'map': tensor(0.3814),
    'map_50': tensor(0.6143),
    'map_75': tensor(0.4557),
    'map_large': tensor(0.5470),
    'map_medium': tensor(0.4010),
    'map_per_class': tensor([0.0000, 0.1002, 0.6347, 0.7505, 0.1525, 0.6505]),
    'map_small': tensor(0.0037),
    'mar_1': tensor(0.2909),
    'mar_10': tensor(0.3833),
    'mar_100': tensor(0.3833),
    'mar_100_per_class': tensor([0.0000, 0.1000, 0.6500, 0.7500, 0.1500, 0.6500]),
    'mar_large': tensor(0.5500),
    'mar_medium': tensor(0.4000),
    'mar_small': tensor(0.0071)}
=================================================
TestScenario.Offline qps=1.81, mean=0.7691, time=1.107, acc=0.000%, mAP=38.140%, mAP_classes={'Props': 0.0, 'TrafficSign': 0.10024752467870712, 'Car': 0.6346888542175293, 'Van': 0.7504950761795044, 'Pedestrian': 0.1524752527475357, 'Truck': 0.6504950523376465}, queries=2, tiles=50.0:0.7691,80.0:0.7713,90.0:0.7721,95.0:0.7725,99.0:0.7728,99.9:0.7728
======================================================================

```


Please check [CM CUDA docs](README-cuda.md) to benchmark ABTF model using MLPerf loadgen.



## Using Docker

We prepared a demo to use Docker with CM automation for ABTF.
You can find and run scripts to build and run Docker (CPU & CUDA) with CM 
[here](https://github.com/mlcommons/cm4abtf/tree/dev/docs/test-abtf-model/docker).

You can then use above commands to benchmark ABTF model.
Don't forget to set up your private GitHub tocken to be able 
to pull a [private GitHub repo with ABTF model](https://github.com/mlcommons/abtf-ssd-pytorch/tree/cognata

```bash
export CM_GH_TOKEN="YOUR TOKEN"
```






## Feedback

Join MLCommons discord or get in touch with developers: Radoyeh Shojaei and Grigori Fursin

