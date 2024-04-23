Examples:


```bash
cmr "get raw dataset mlcommons-cognata" --import=${HOME}/datasets/cognata -j
cmr "get raw dataset mlcommons-cognata" --import=%userprofile%\datasets\cognata -j
cmr "get raw dataset mlcommons-cognata" --import=D:\Work2\cognata -j
```


```bash
cmr "get raw dataset mlcommons-cognata" --path=${HOME}/datasets/cognata -j
cmr "get raw dataset mlcommons-cognata" --path=%userprofile%\datasets\cognata -j
cmr "get raw dataset mlcommons-cognata" --path=D:\Work2\cognata-downloaded -j

cm show cache --tags=dataset,mlcommons-cognata

cm rm cache --tags=dataset,mlcommons-cognata
```

```bash
cmr "get raw dataset mlcommons-cognata" --serial_numbers=10002_Urban_Clear_Morning
cmr "get raw dataset mlcommons-cognata" --serial_numbers=10002_Urban_Clear_Morning --group_names=Cognata_Camera_02_8M
```


