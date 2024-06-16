# Run the POC Reference

## Install Dependencies

MLCommons CM Automation framework is used to run the POC reference. CM requires Python 3.7+, git, python3-pip and python3-venv. If these dependencies are present you can do

Activate a venv for CM (Not mandatory but recommended)
```
python3 -m venv cm
source cm/bin/activate
```

Install CM and pull the needed repositories
```
pip install cm4abtf
```

More installation details can be found at [CM Installation Page](https://docs.mlcommons.org/ck/install)

Using an Ubuntu example, run `cm` in the terminal and if CM successfully installed, expect the following output:

```
(cm) user@ubuntu:~$ cm 
 cm {action} {automation} {artifact(s)} {--flags} @input.yaml @input.json
```

Note: the `(cm)` indicates that Python `venv` is active.

The computer running the POC reference needs Docker. Follow instructions described in the link below:

* [Install Docker Engine](https://docs.docker.com/engine/install/)

Note: if you're running Ubuntu, CM automatically installs Docker when running the benchmark.


Now, the `cm` cli commands will be used to run the POC reference.

## Start the Benchmark

By running the script below, you are downloading the POC container and dataset, then launching the benchmark. All in one command!

```
cm run script --tags=run-abtf,_poc-demo \
  --quiet \
  --docker \
  --gh_token=<GH_TOKEN> \
  --docker_cache=no
```
!!! tip
    * Use `--rerun` to force overwrite the previously generated results
    * Use `--env.CM_MLPERF_LOADGEN_BUILD_FROM_SRC=off` to use the prebuilt MLPerf Loadgen binary and not do a source compilation
    * Use `--docker_os=[rhel|arch|ubuntu]` to change the docker OS
    * Use `--docker_os_version=[8|9]` for `RHEL`, `[24.04,22.04,20.04]` for `ubuntu` and `[latest]` for `arch`  
    * Use `--docker_base_image=[IMAGE_NAME]` to override the default base image for docker
    * Skip `--docker` to do the run on the host machine without using a docker
    * Github actions for this run can be seen [here](https://github.com/mlcommons/cm4abtf/actions/workflows/test-mlperf-inference-abtf-poc.yml)

Depending on the computer used and internet connection, this can take a few minutes.

## Results

Once the benchmark successfully has run, KPIs should print in the terminal window and the dataset frames should have been labeled.

### KPIs

For the POC only the average latency and accuracy is measured. This will change ... 

### Labeled Frames

In the following directory... 

IMG Before

IMG After (w. bounding boxes)

## Contact

If you face any issues, please don't hesitate to reach out!

* Raise a [GitHub Issue](https://github.com/mlcommons/cm4abtf/issues)
* Join the [MLCommons Automotive Discord Server](https://discord.gg/jBxH9GvftZ)

## Contributors
* The POC reference model is trained and developed by Radoyeh Shojaei
* MLPerf Loadgen integration for the POC reference is done by Grigori Fursin and Radoyeh Shojaei
* CM workflow for the POC reference is done by Arjun Suresh and Grigori Fursin
