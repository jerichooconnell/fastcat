## Installation

Most users should just install the program locally
`pip install setup.py`

This document gives some additional details only interesting for some users.

* [Requisites](#requisites)
	* [Install a scientific python distribution](#install-a-scientific-python-distribution)
	* [Install python and the packages](#install-python-and-the-packages)
* [Download and run](#download-and-run)

## Requisites

### YOU NEED CUDA to install TIGRE t install Fastcat
Good luck! Installing CUDA is the worst. You would think that the package made by the multi billion dollar company would be easy to install but it is not, at least on linux.

You need python (3) with the following packages to run this program:
```
matplotlib, numpy, scipy, XlsxWriter, tigre, astropy, tk
```
The last one is only needed if you want to make use of the GUI.

You can [install a scientific python distribution](#install-a-scientific-python-distribution) providing them or you can install [only what you need](#install-python-and-the-packages). The first is recommended for Windows users, the latter for Linux users.

### Installing Tigre


### Install a scientific python distribution:
For example you can try [Anaconda](https://www.continuum.io/downloads).
Since this will install pip, you might want to use the [pip installer](#installation).

### Install python and the packages:
- Download [python](https://www.python.org/) for your OS. See specific instructions to install from repositories below.
- Use the [pip installer](#Installation) (recommended) or manually install the additional packages. See specific instructions below for installs based on python 3.X.