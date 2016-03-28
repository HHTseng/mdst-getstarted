# Deep Learning with Torch

A practical beginner-level tutorial and some code to get you started
with Convolutional Neural Networks.

## Quick-start Guide

#### 0. Get setup on Flux

If you haven't already, take a look at our
[Intro to Flux Tutorial](https://github.com/MichiganDataScienceTeam/mdst-getstarted/tree/master/flux).
Before getting started with Torch, make sure you are able to do all of the
following:

1. Log in to Flux
2. Spawn an interactive job on the `mdatascienceteam_flux` allocation
3. Clone this repository into your scratch directory. You should have the repo stored here:

```
$ TUTORIAL_DIR=/scratch/mdatascienceteam_flux/your_uniqname/mdst-getstarted/tutorials/torch
```

#### 1. Download the MNIST dataset

For convenience, we've included a script that will download and unpack
the MNIST dataset for you. From this directory, navigate to the
`mnist` directory and run the script as follows:

```
$ cd $TUTORIAL_DIR/data/mnist
$ ./download.sh
```

#### 2. Start an interactive job

See the [flux tutorial](http://torch.ch/docs/getting-started.html) for
a guide on running jobs on Flux.

#### 3. Load Torch

A working version of Torch is already installed on Flux. To access it,
simply load the `torch/latest' module as follows:

```
$ module load lsa torch/latest
```

If you'd prefer to use Torch on your local machine (not recommended),
you'll first need to install Torch. You can find detailed instructions
on the Torch website
[here](http://torch.ch/docs/getting-started.html). For GPU support,
you'll need to install
[a few additional things](https://github.com/torch/torch7/wiki/Cheatsheet#gpu-support).


#### 4. Run the training script

To train your first CNN, navigate to the `src` directory and run the
`main.lua` script from Torch.

```
$ cd $TUTORIAL_DIR/src
$ th main.lua -nTrain 1000 -validFrac 0.5
```

Congrats! You're now training a deep Convolutional Neural Network for
handwritten digit recognition. Since you're only using 500 training
samples (The MNIST dataset has 60000 in total), don't expect to get
state-of-the-art accuracy just yet.

#### 5. Evaluate your model

TODO

## Convolutional Neural Networks (CNNs)

TODO

## Training and Evaluation Options

TODO

## Specifying New Network Architectures

TODO

## Importing Datasets

TODO