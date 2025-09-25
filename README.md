# Find Your Course TD + Exercices

## Description

This is a PyTorch implementation of a deep convolutional neural network model trained on Places365 data. The model is trained on a subset of 100K images which have outcome labels that are associated to factors which are relevant for environmental health.

📌 Take a look `Material` folder after reading this README, there is where your journey begins !

## Learning Outcomes

- Be aware of different types of Computer Vision tasks
- Load an image dataset in PyTorch
- Be able to explain what a convolutional layer does and how it's different from a fully-connected layer
- Identify different components of a CNN
- Load a pre-trained model in PyTorch
- Be able to use PyTorch to train a model on a dataset
- Iterate on design choices for model training

## Requirements

### Academic

- Basic familiarity with Python & basic programming skills is required 
- Familiarity with Jupyter Notebooks is recommended 
- Curiosity and willingness to learn !

### System

| Program                                                    | Version                  |
| ---------------------------------------------------------- | ------------------------ |
| [Python](https://www.python.org/downloads/)                | >= 3.9                   |
| [Miniconda or anaconda](https://www.anaconda.com/products/distribution) | latest                   |


## Getting started

Clone this repository into your local drive.

```sh
git clone https://github.com/magicAyyub/FYC.git
cd FYC
```

### Setting up a virtual environment

We will set up a virtual environment for running our scripts. In this case, installing specific package versions will not interfere with other programmes we run locally as the environment is contained. Initially, let's set up a virtual environment:

`If you have cuda gpu on your machine, decomment line 10 in environment.yml`

```sh
conda env create -f environment.yml
```

This will create a new folder for the virtual environment named `perceptions` in your repository. We activate this environment by running

```sh
conda activate fyc_env
```

All the dependencies are installed along with the virtual environment. We will manually install the development tools since we do not need those dependencies when we export to HPC and create a virtual environment there.

### Setting up the development virtual environment

The `pytest` and pre-commit module is required for running tests and formatting. This can be installed by running:

```sh
conda install --file requirements-dev.txt
```

Now run the tests below to make sure everything is set up correctly. Then, proceed to the video.

### Testing

To run all tests, install `pytest`. After installing, run

```sh
pytest tests/ -v
```