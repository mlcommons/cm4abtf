# Issues, notes and prototypes

### Grigori: weird case on Windows

If I download baseline_8mp_ss_scales_ep15.pth to the ROOT directory with the virtual environment,
pip stops working since it considers this file as a broken package ...


### Grigori: misc commands

Register local ABTF model in CM cache to be the default

```bash
cmr "get ml-model abtf-ssd-pytorch _local.baseline_8mp_ss_scales_ep15.pth"
```

### Grigori: benchmarking other models

Other ways to download public or private model code and weights:
```bash
cmr "get ml-model abtf-ssd-pytorch _skip_weights" --adr.abtf-ml-model-code-git-repo.env.CM_ABTF_MODEL_CODE_GIT_URL=https://github.com/mlcommons/abtf-ssd-pytorch
cmr "get ml-model abtf-ssd-pytorch _skip_weights" --model_code_git_url=https://github.com/mlcommons/abtf-ssd-pytorch --model_code_git_branch=cognata-cm
cmr "get ml-model abtf-ssd-pytorch _skip_weights _skip_code"
```

Other ways to run local (private) model:

You can first copy ABTF model code from GitHub to your local directory `my-model-code`.

```bash
cmr "generic loadgen python _pytorch _custom _cmc" --samples=5 --modelsamplepath=0000008766.png.cpu.pickle \
  --modelpath=baseline_8mp_ss_scales_ep15.pth \
  --modelcfg.num_classes=13 \
  --modelcodepath="my-model-code" \
  --modelcfg.config=baseline_8MP_ss_scales
```

### Grigori: import Cognata dataset from local folder

```bash

cmr "get raw dataset mlcommons-cognata" --import=D:\Work2\cognata
```