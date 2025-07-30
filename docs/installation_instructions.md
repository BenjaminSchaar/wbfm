
# Installation for developing

### Install Anaconda/Mamba

We suggest installing Mamba:
https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html

The official page is here, but please be wary of the terms of service:
https://www.anaconda.com/products/individual

### Get the code

Download or clone the repositories that you want to modify; most likely just the top one:
1. wbfm (this repo): https://github.com/Zimmer-lab/wbfm
2. centerline_behavior_annotation: https://github.com/Zimmer-lab/centerline_behavior_annotation 
3. imutils: https://github.com/Zimmer-lab/imutils

Note that you will have to install all of them later; you can either clone and install locally (instructions below) or install via pip using for example:

```commandline
pip install git+[url here]
```


### Install the environments

#### Pre-installed environments

Note: there are pre-installed environments living on the cluster, at:
/lisc/scratch/neurobiology/zimmer/.conda/envs/wbfm

They can be activated using:
```commandline
conda activate /lisc/scratch/neurobiology/zimmer/.conda/envs/wbfm
```

#### Creating new environments

Note: if you are just using the GUI, then you can use a simplified environment.
Detailed instructions can be found in the [README](wbfm/gui/README.md) file under the gui section
For running the full pipeline you need the environment found here:

1. conda-environments/wbfm_dev.yaml

This installs the public packages, now we need to install our local libraries.
Do `conda activate wbfm` (or whatever your name is) and install the local code in the following way:

1. cd to the repository
2. run: `pip install -e .`
3. Repeat steps 1-2 for the other repositories


If you do not need a local clone for easy modification, you can install these through github directly using:
```commandline
pip install git+[url here]
```


#### Summary of installations

You will install 4 "things": 1 conda environment (from the yaml file) and 3 custom packages
