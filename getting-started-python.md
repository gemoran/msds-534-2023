# Getting started with Python

## Installation

Download Anaconda [here](https://www.anaconda.com/download).

(Note: there are a variety of different python distributions; for statistics and machine learning, we recommend Anaconda.)

## Managing packages

Similar to `R`, there are many open source Python packages for statistics and machine learning.

To download packages, two popular package managers are `pip` and `conda`.  Both `pip` and `conda` come with the Anaconda distribution. 

## Environments

### About

We recommend using virtual environments with Python. From [this blog](https://www.dataquest.io/blog/a-complete-guide-to-python-virtual-environments/):

> A Python virtual environment consists of two essential components: the Python interpreter that the virtual environment runs on and a folder containing third-party libraries installed in the virtual environment. These virtual environments are isolated from the other virtual environments, which means any changes on dependencies installed in a virtual environment don’t affect the dependencies of the other virtual environments or the system-wide libraries. Thus, we can create multiple virtual environments with different Python versions, plus different libraries or the same libraries in different versions.

![](https://www.dataquest.io/wp-content/uploads/2022/01/python-virtual-envs1.webp)

### Creating an environment for MSDS-534

We recommend creating a virtual environment for your MSDS-534 coding projects.

1. Open Terminal
2. Create an environment called `msds534` using `conda` with the command:
   ```conda create --name msds534```
3. To install packages in your environment, first activate your environment:
   ```conda activate msds534```
4. Then, install the following packages using the command:
   ```conda install numpy pandas matplotlib seaborn scikit-learn```
5. Install PyTorch by running the appropriate command from [here](https://pytorch.org) (for macOS, the command is: `conda install pytorch::pytorch torchvision torchaudio -c pytorch`
6. To exit your environment:
   ```conda deactivate```

Here is a helpful [cheatsheet](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) for `conda` environment commands.

