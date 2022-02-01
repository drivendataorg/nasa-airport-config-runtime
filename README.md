# Run-way Functions: Prediction Reconfigurations at US Airports

![Python 3.9](https://img.shields.io/badge/Python-3.9-blue) [![GPU Docker Image](https://img.shields.io/badge/Docker%20image-gpu--latest-green)](https://hub.docker.com/r/drivendata/nasaairportconfig-competition/tags?page=1&name=gpu-latest) [![CPU Docker Image](https://img.shields.io/badge/Docker%20image-cpu--latest-green)](https://hub.docker.com/r/drivendata/nasaairportconfig-competition/tags?page=1&name=cpu-latest)


### For instructions about how to submit to the [Run-way Functions Challenge](https://www.drivendata.org/competitions/92/competition-nasa-airport-configuration-prescreened), start with the [Code submission format page](https://www.drivendata.org/competitions/92/competition-nasa-airport-configuration-prescreened/page/442/) of the competition website.

Welcome to the runtime repository for the [Run-way Functions: Predict Reconfigurations at US Airports](https://www.drivendata.org/competitions/92/competition-nasa-airport-configuration-prescreened/). This repository contains the definition of the environment where your code submissions will run. It specifies both the operating system and the software packages that will be available to your solution.

This repository has three primary uses for competitors:

:bulb: **Provide example solutions**: You can find two examples to help you develop your solution. 
1. [Baseline solution](https://github.com/drivendataorg/nasa-airport-config-runtime/tree/main/submission_src): minimal code that runs succesfully in the runtime environment output and outputs a proper submission. This simply predicts equal probabilities for each configuration at an airport. You can use this as a guide to bring in your model and generate a submission.
2. <!--TODO: implement or remove-->Implementation of the [PyTorch benchmark](https://github.com/drivendataorg/cloud-cover-runtime/tree/main/benchmark_src): submission code based on the [benchmark blog post](https://www.drivendata.co/blog/cloud-cover-benchmark/)

:wrench: **Test your submission**: Test your `submission` files with a locally running version of the container to discover errors before submitting to the competition site. You can also find an [evaluation script](https://github.com/drivendataorg/nasa-airport-config-runtime/blob/main/runtime/scripts/metric.py) for implementing the competition metric.

:package: **Request new packages in the official runtime**: Since the Docker container will not have network access, all packages must be pre-installed. If you want to use a package that is not in the runtime environment, make a pull request to this repository. Make sure to test out adding the new package to both official environments, [CPU](https://github.com/drivendataorg/nasa-airport-config-runtime/blob/main/runtime/environment-cpu.yml) and [GPU](https://github.com/drivendataorg/nasa-airport-config-runtime/blob/main/runtime/environment-gpu.yml).

----


### [0. Getting started](#getting-started)
 - [Prerequisites](#prerequisites)
 - [Development dataset](#development-dataset)
 - [Submission format](#submission-format)
### [1. Testing a submission locally](#testing-a-submission-locally)
 - [Running your submission locally](#running-your-submission-locally)
 - [Scoring your predictions](#scoring-your-predictions)
 - [Running the benchmark](#running-the-benchmark)
### [2. Runtime network access](#runtime-network-access)
### [3. Troubleshooting](#troubleshooting)
### [4. Updating runtime packages](#updating-runtime-packages)

----

## Getting started

### Prerequisites

 - A clone or fork of this repository
 - [Docker](https://docs.docker.com/get-docker/)
 - At least ~12 GB of free space for both the sample data and Docker images
 - [GNU make](https://www.gnu.org/software/make/) (optional, but useful for running the commands in the Makefile)

Additional requirements to run with GPU:

 - [NVIDIA drivers](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#package-manager-installation) with **CUDA 11**
 - [NVIDIA Docker container runtime](https://nvidia.github.io/nvidia-container-runtime/)

### Development dataset

To run a submission, the code execution environment reads features and the submission format from the `runtime/data` directory on the host machine.

```
$ tree runtime/data
├── katl
│	├── katl_awefwef.csv.bz2
│	└── ...
├── kclt
│	├── kclt_awefwef.csv.bz2
│	└── ...
├── ksea
│	├── kclt_awefwef.csv.bz2
│	└── ...
└── final_submission_format.csv.bz2
```

We do not provide the full prescreened test set, so instead you can use the _development dataset_ to debug your submission; if your submission runs successfully on the development dataset, it should run successfully on the prescreened dataset. The development dataset is created from the open arena training features and labels. (You can download them from the [Data download page](https://www.drivendata.org/competitions/89/competition-nasa-airport-configuration/data/)). It consists of labels from the final day and features from the final day plus a _warm start_ period that starts two days prior to the final day.

The script [`runtime/scripts/generate_development_dataset.py`](https://github.com/drivendataorg/nasa-airport-config-runtime/blob/main/runtime/scripts/generate_development_dataset.py) generates the development dataset. You can learn more about it with the following command:

```
$ python runtime/scripts/generate_development_dataset.py --help
Usage: generate_development_dataset.py [OPTIONS] INPUT_LABELS_PATH
                                       INPUT_FEATURE_DIRECTORY
                                       OUTPUT_DIRECTORY

  Creates a data subset that matches the format of the full evaluation data
  intended for developing your submission.

Arguments:
  INPUT_LABELS_PATH        Path to the training labels.  [required]
  INPUT_FEATURE_DIRECTORY  Directory containing the training features.
                           [required]

  [OUTPUT_DIRECTORY]       Directory where the development dataset will be
                           saved.  [default: runtime/data]



Options:
  --help  Show this message and exit.
```

The default value of `OUTPUT_DIRECTORY` will save the development dataset to `runtime/data`. When testing your submission locally, a Docker container is launched from your computer (or the "host" machine), and the `runtime/data` directory on your host machine is mounted as a read-only directory to `codeexecution/data`. In the runtime, your code will then be able to access test features from `codeexecution/data/test_features`.

For details about how data is accessed in the code execution runtime, see the Code submission format [page](https://www.drivendata.org/competitions/92/competition-nasa-airport-configuration-prescreened/page/442/).

### Code submission format

Time is a key element in this competition―we're interested in a _real-time_ solution, one that can predict the future using only information available at the present. Assuring that a solution doesn't (accidentally or otherwise) use information from the future is complicated since the final evaluation dataset is a static dataset containing many different prediction times; a feature at 9 AM is valid for predicting 10 AM (it's in the past) but invalid for predicting 8 AM (it's in the future). In other words, each prediction time defines a unique set of valid features, different from that of all other prediction times! This makes it challenging to ensure that your submission only uses valid time points for each prediction time (and even more challenging for the competition hosts to validate that _all_ submissions are valid!).

The code execution runtime is designed to avoid the need to track valid and invalid features. It simulates real-time conditions for each prediction time:

1. The [**supervisor**](https://github.com/drivendataorg/nasa-airport-config-runtime/blob/submission-service-alternating/runtime/supervisor.py) script creates a time-censored extract of the features, i.e., features _prior to_ the prediction time, and stores them to `/codeexecution/data` with the same directory structure as the training features. It also creates a **partial submission format**, `/codeexecution/data/partial_submission_format.csv`, which includes only rows for the current prediction time.
2. Your `main.py` runs with a single argument, the prediction time, e.g., `2022-06-09T14:00:00`. It can read your model assets, any or all of the time-censored features, and any intermediate files you may have written out in previous iterations.. It must write a CSV to `/codeexecution/data/prediction.csv` with the same format as the partial submission format, with your predicted probabilities in the `active` column.
3. A utility script checks that your predictions for this prediction time has the same indices and same columns as the partial submission format and that the probabilities for each airport and lookahead sum to 1.

The process is repeated for each prediction time in order, so as long as you only read from `/codeexecution/data`, you can be sure your model isn't using information from the future.

**Do**s and **don't**s

**Do**:
- Read features from `/codeexecution/data`. These are guaranteed to be in the past relative to the prediction time.
- Write out and read intermediate files. Just be careful not to use any of the reserved file locations: `/codeexecution/prediction.csv`, `/codeexecution/submission.csv`, `/codeexecution/predictions`.

**Don't**:
- Read from locations other than `/codeexecution/data`, your model assets, and intermediate files that your submission produces.

We provide the supervisor and utility scripts; all you need to do is provide `main.py` that implements the 

**supervisor** script simulates a real-time operational environment, curating a directory of features that is guaranteed to only contain events from before the prediction time. For each prediction time, the supervisor process generates a directory of time-censored features, 

Your submission is a script `main.py` that implements a command line interface taking a single positional argument, the prediction time, e.g., `2022-06-09T14:00:00`. If we imagine your solution running in real-time, the prediction time would be the current time, and your model would generate predictions for how each of the airports will be configured from 30 minutes to 6 hours into the future. As such, at each prediction time, your model has access to features time-censored features, i.e., features _prior to_ the prediction time. The earliest prescreened feature begins two days prior to the first prediction time. (The first prediction time is on 2021-10-17; the first feature time is 2021-10-15 00:00:00.) At each iteration, the appropriate time-censored input features will be stored in a fixed location `/codeexecution/data` with the same directory structure as the training features. The directory also contains a partial submission format, `/codeexecution/data/partial_submission_format.csv`,  that only includes rows for the current prediction time. The output at each iteration should be a CSV with the same format as the partial submission format, with your predictions in the `active` column. In short, the `/codeexecution/data` directory contains all the data your script needs to make a valid prediction for a single timestamp, e.g.:


## Testing a submission locally

Your submission will run inside a Docker container, a virtual operating system that allows for a consistent software environment across machines. **The best way to make sure your official submission to the DrivenData site will run is to first run it successfully in the container on your local machine.**

On the official code execution platform, the test features and test metadata will already be mounted. The root level of your submission must contain a `main.py`. The steps that take place in the code execution platform are:

1. Extract your submission and check that it contains a file `main.py`.
2. Get a list of the prediction times.
3. For each prediction time:
  




1. Run `main.py` to generate predictions. `main.py` must implement a command line interface (CLI) that perform inference on all of the test chips in `/codeexecution/data/test_features` and write predictions in the form of single-band 512x512 TIFs into the `/codeexecution/predictions` folder

2. Compress all of the TIFs from `codeexecution/predictions` into a tar archive. The tar archive is then sent out for scoring - it is not scored inside of the code execution platform.

> **Note:** <!-- TODO: explain how they might have access to `/data` in the local runtime but will not in the real deal -->

For the full requirements of a submission, see the [Code submission format page](https://www.drivendata.org/competitions/92/competition-nasa-airport-configuration-prescreened/page/442/).

### Running your submission locally

This section provides instructions on how to run the your submission in the code execution container from your local machine. To simplify the steps, key processes have been defined in the `Makefile`. Commands from the `Makefile` are then run with `make {command_name}`. The basic steps are:
```
make pull
make pack-submission
make test-submission
```

1. Set up the [prerequisites](#prerequisites)

2. Save [fake data](#fake-test-data) in `runtime/data`

3. Download the official competition Docker image:

    ```bash
    $ make pull
    ```

4. Save all of your submission files, including at least the required `main.py` script, in the `submission_src` folder of the runtime repository. Make sure any needed model weights are saved in `submission_src` as well.

5. Create a `submission/submission.zip` file containing your code and model assets:
    ```bash
    $ make pack-submission
    cd submission_src; zip -r ../submission/submission.zip ./*
      adding: main.py (deflated 50%)
    ```

6. Launch an instance of the competition Docker image, and run the same inference process that will take place in the official runtime:
   ```
   $ make test-submission
   ```
    
    This unzips `submission/submission.zip` in the root directory of the container, and then runs `main.py`. The resulting prediction TIFs in `codeexecution/predictions/` are then compressed into a tar archive for scoring. The tar archive is saved out to `submission/submission.tar.gz` on your local machine.
   
> Remember that `codeexecution/data/test_features` is a mounted version of what you have saved locally in `runtime/data/test_features`. In the official code execution platform, `codeexecution/data/test_features` will contain the actual test features.

When you run `make test-submission` the logs will be printed to the terminal and written out to `submission/log.txt`. If you run into errors, use the `log.txt` to determine what changes you need to make for your code to execute successfully. For an example of what the logs look like when the full process runs successfully, see [`example_log.txt`](https://github.com/drivendataorg/cloud-cover-runtime/blob/main/example_log.txt).

### Scoring your predictions

We have provided a [metric script](https://github.com/drivendataorg/nasa-airport-config-runtime/blob/main/runtime/scripts/metric.py) to calculate the competition metric in the same way scores will be calculated in the DrivenData platform. To score your submission:

1. After running the above, the predictions generated by your code should be saved in an archive at `submission/submission.tar.gz`. Unzip your submission into `submission/predictions`:
   ```bash
   $ mkdir submission/predictions
   $ tar -xf submission/submission.tar.gz --directory submission/predictions 
   ```

2. Make sure the labels for your fake test data are saved in `runtime/data/test_labels` in the same format as the training labels. For example, if you have a chip with id `abcd` in `runtime/data/test_features`, the label for that chip should be saved at `runtime/data/test_labels/abcd.tif`
   
3. Run `runtime/scripts/metric.py` on your predictions:
    ```bash
    # show usage instructions
    $ python runtime/scripts/metric.py --help
    Usage: metric.py [OPTIONS] SUBMISSION_DIR ACTUAL_DIR

      Given a directory with the predicted mask files (all values in {0, 1}) and
      the actual mask files (all values in {0, 1}), get the overall
      intersection-over-union score

    Arguments:
      SUBMISSION_DIR  [required]
      ACTUAL_DIR      [required]

    Options:
      --install-completion  Install completion for the current shell.
      --show-completion     Show completion for the current shell, to copy it or
                            customize the installation.

      --help                Show this message and exit.

    # run script on your predictions
    $ python runtime/scripts/metric.py submission/predictions runtime/data/test_labels
    2021-12-14 12:42:06.112 | INFO     | __main__:main:42 - calculating score for 2 image pairs ...
    100%|█████████████████████████| 2/2 [00:00<00:00, 293.04it/s]
    2021-12-14 12:42:06.140 | SUCCESS  | __main__:main:44 - overall score: 0.5
    ```

### Running the benchmark

The code for the [PyTorch benchmark](https://www.drivendata.co/blog/cloud-cover-benchmark/) is provided to demonstrate how a correct submission can be structured. See the benchmark [blog post](https://www.drivendata.co/blog/cloud-cover-benchmark/) for a full walkthrough. The process to run the benchmark is the same as running your own submission, except that you will reference code in `benchmark_src` rather than `submission_src`.

To run the benchmark submission locally:

1. Set up the [prerequisites](#prerequisites)

2. Save [fake data](#fake-test-data) in `runtime/data`

3. Download the official competition Docker image:

    ```bash
    $ make pull
    ```

4. Compress the files in `benchmark_src` to `submission/submission.zip`:
   ```
   $ make pack-benchmark
   cd benchmark_src; zip -r ../submission/submission.zip ./*
    adding: assets/ (stored 0%)
    adding: assets/cloud_model.pt (deflated 7%)
    adding: cloud_dataset.py (deflated 63%)
    adding: cloud_model.py (deflated 74%)
    adding: losses.py (deflated 57%)
    adding: main.py (deflated 64%)
   ```
   To avoid losing your work, this command will not overwrite an existing submission. To generate a new submission, you will first need to remove the existing `submission/submission.zip`.

5. Launch an instance of the competition Docker image, and run the same inference process that will take place in the official runtime:
   ```
   $ make test-submission
   ```
   Just like with your submission, the final predictions will be compressed into a tar archive and saved to `submission/submission.tar.gz` on your local machine.

### Downloading pre-trained weights

It is common for models to download pre-trained weights from the internet. Since submissions do not have open access to the internet, you will need to include all weights along with your `submission.zip` and make sure that your code loads them from disk and rather than the internet.

For example, PyTorch uses a local cache which by default is saved to `~/.cache/torch`. Identify which of the weights in that directory are needed to run inference (if any), and copy them into your submission. If we need pre-trained ResNet34 weights we downloaded from online, we could run:
```sh
# Copy your local pytorch cache into submission_src/assets
cp ~/.cache/torch/checkpoints/resnet34-333f7ec4.pth submission_src/assets/

# Zip it all up in your submission.zip
zip -r submission.zip submission_src
```

If we wanted to copy all of the contents in the PyTorch cache, we could instead run `cp -R ~/.cache/torch submission_src/assets/`. When the platform runs your code, it will extract `assets` to `/codeexecution/assets`. You'll need to tell PyTorch to use your custom cache directory instead of `~/.cache/torch` by setting the `TORCH_HOME` environment variable in your Python code (in `main.py` for example).

```python
import os
os.environ["TORCH_HOME"] = "/codeexecution/assets/torch"
```

Now PyTorch will load the model weights from the local cache, and your submission will run correctly in the code execution environment without downloading from the internet.

## Troubleshooting

#### CPU and GPU

The `make` commands will try to select the CPU or GPU image automatically by setting the `CPU_OR_GPU` variable based on whether `make` detects `nvidia-smi`.

**If you have `nvidia-smi` and a CUDA version other than 11**, you will need to explicitly set `make test-submission` to run on CPU rather than GPU. `make` will automatically select the GPU image because you have access to GPU, but it will fail because `make test-submission` requires CUDA version 11. 
```bash
CPU_OR_GPU=cpu make pull
CPU_OR_GPU=cpu make test-submission
```

If you want to try using the GPU image on your machine but you don't have a GPU device that can be recognized, you can use `SKIP_GPU=true`. This will invoke `docker` without the `--gpus all` argument.

## Updating runtime packages

If you want to use a package that is not in the environment, you are welcome to make a pull request to this repository. If you're new to the GitHub contribution workflow, check out [this guide by GitHub](https://docs.github.com/en/get-started/quickstart/contributing-to-projects). The runtime manages dependencies using [conda](https://docs.conda.io/en/latest/) environments. [Here is a good general guide](https://towardsdatascience.com/a-guide-to-conda-environments-bc6180fc533) to conda environments. The official runtime uses **Python 3.9.7** environments.

To submit a pull request for a new package:

1. Fork this repository.
   
2. Edit the [conda](https://docs.conda.io/en/latest/) environment YAML files, `runtime/environment-cpu.yml` and `runtime/environment-gpu.yml`. There are two ways to add a requirement:
    - Add an entry to the `dependencies` section. This installs from a conda channel using `conda install`. Conda performs robust dependency resolution with other packages in the `dependencies` section, so we can avoid package version conflicts.
    - Add an entry to the `pip` section. This installs from PyPI using `pip`, and is an option for packages that are not available in a conda channel.

    For both methods be sure to include a version, e.g., `numpy==1.20.3`. This ensures that all environments will be the same.

3. Locally test that the Docker image builds successfully for CPU and GPU images:

    ```sh
    CPU_OR_GPU=cpu make build
    CPU_OR_GPU=gpu make build
    ```

4. Commit the changes to your forked repository.
   
5. Open a pull request from your branch to the `main` branch of this repository. Navigate to the [Pull requests](https://github.com/drivendataorg/nasa-airport-config-runtime/pulls) tab in this repository, and click the "New pull request" button. For more detailed instructions, check out [GitHub's help page](https://help.github.com/en/articles/creating-a-pull-request-from-a-fork).
   
6. Once you open the pull request, Github Actions will automatically try building the Docker images with your changes and running the tests in `runtime/tests`. These tests can take up to 30 minutes, and may take longer if your build is queued behind others. You will see a section on the pull request page that shows the status of the tests and links to the logs.
   
7. You may be asked to submit revisions to your pull request if the tests fail or if a DrivenData team member has feedback. Pull requests won't be merged until all tests pass and the team has reviewed and approved the changes.

---

## Good luck; have fun!

Thanks for reading! Enjoy the competition, and [hit up the forums](https://community.drivendata.org/c/runway/52) if you have any questions!
