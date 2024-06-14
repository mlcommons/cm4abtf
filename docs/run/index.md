# Run the POC Reference

## Install Dependencies

MLCommons CM Automation framework is used to run the POC reference. CM needs to be installed on the computer running the benchmark and CM requires Python 3.7+. Please follow the instructions to install CM as described in the link below:

* [CM Installation Page](https://docs.mlcommons.org/ck/install)

Using an Ubuntu example, run `cm` in the terminal and if CM successfully installed, expect the following output:

```
(cm) user@ubuntu:~$ cm 
 cm {action} {automation} {artifact(s)} {--flags} @input.yaml @input.json
```

Note: the `(cm)` indicates that Python `venv` is active.

The computer running the POC reference needs Docker. Follow instructions described in the link below:

* [Install Docker Engine](https://docs.docker.com/engine/install/)

Note: if you're running Ubuntu, CM automatically installs Docker when running the benchmark.

## Download Repositories

Start by using CM to pull the necessary repositories.

```bash
cm pull repo gateoverflow@cm4mlops
cm pull repo mlcommons@cm4abtf
```

Now, the `cm` cli commands will be used to run the POC reference.

## Start the Benchmark

By running the script below, you are downloading the POC container and dataset, then launching the benchmark. All in one command!

```
cm run script --tags=run-abtf,_poc-demo --quiet --docker --gh_token=<GH_TOKEN>  --docker_cache=no
```

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

TBD