# Advanced installation

Most users should just install the program using
`pip install xpecgen`.

This document gives some additional details only interesting for some users.

* [Requisites](#requisites)
	* [Install a scientific python distribution](#install-a-scientific-python-distribution)
	* [Install python and the packages](#install-python-and-the-packages)
* [Download and run](#download-and-run)

## Requisites
You need python (either 2 or 3) with the following packages to run this program:
```
matplotlib, numpy, scipy, XlsxWriter, tk
```
The last one is only needed if you want to make use of the GUI.

You can [install a scientific python distribution](#install-a-scientific-python-distribution) providing them or you can install [only what you need](#install-python-and-the-packages). The first is recommended for Windows users, the latter for Linux users.


### Install a scientific python distribution:
For example you can try [Anaconda](https://www.continuum.io/downloads).
Since this will install pip, you might want to use the [pip installer](#installation).

### Install python and the packages:
- Download [python](https://www.python.org/) for your OS. See specific instructions to install from repositories below.
- Use the [pip installer](#Installation) (recommended) or manually install the additional packages. See specific instructions below for installs based on python 3.X.

#### Windows
As a general advice, forget it. Scipy depends on lots of C, Cython and Fortran code that needs to be compiled before use.
I suggest you just go for [a scientific python distribution](#install-a-scientific-python-distribution).

There are also some alternatives which are actually Linux in disguise:
- In Windows 10 you can make use of the bash shell to install from the Ubuntu repositories. Check out [this guide](http://www.howtogeek.com/249966/how-to-install-and-use-the-linux-bash-shell-on-windows-10/) and the Ubuntu specific instructions. (I have not tested this yet).
- Install Ubuntu (or any other Linux) in a virtual machine using [VirtualBox](https://www.virtualbox.org/).
- [Switch to Linux](https://www.google.com/search?q=why+switch+to+linux).

#### Ubuntu
```bash
sudo apt-get update
sudo apt-get install python3 python3-matplotlib python3-numpy python3-scipy python3-xlsxwriter python3-tk
```
#### Arch Linux
```bash
sudo pacman -S python python-matplotlib python-numpy python-scipy python-xlsxwriter tk
```
#### Fedora
(Not tested)
```bash
sudo yum install -y python-pip
sudo yum install -y lapack lapack-devel blas blas-devel 
sudo yum install -y blas-static lapack-static
sudo pip install numpy
sudo pip install scipy
sudo pip install openpyxl
```
On Fedora 23 onwards, use dnf instead of yum

## Download and run
You can also download and execute the program without installing it, as long as you meet the [requisites](#requisites).
Download and extract the [zip file](https://github.com/Dih5/xpecgen/archive/master.zip) of the repository.
To start the GUI, open xpecgen/xpecgenGUI.py  as a package with your python interpreter: 
```bash
python -m xpecgen.xpecgenGUI
```