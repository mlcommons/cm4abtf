# Issues

## Weird case on Windows

### 20240410: Grigori

If I download baseline_8mp_ss_scales_ep15.pth to the ROOT directory with the virtual environment,
pip stops working since it considers this file as a broken package ...


# Misc commands

Register local ABTF model in CM cache to be the default

```bash
cmr "get ml-model abtf-ssd-pytorch _local.baseline_8mp_ss_scales_ep15.pth"
```
