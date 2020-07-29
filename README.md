# xpecgen
[![Github release](https://img.shields.io/github/release/dih5/xpecgen.svg)](https://github.com/dih5/xpecgen/releases/latest)
[![PyPI](https://img.shields.io/pypi/v/xpecgen.svg)](https://pypi.python.org/pypi/xpecgen)
[![Semantic Versioning](https://img.shields.io/badge/SemVer-2.0.0-brightgreen.svg)](http://semver.org/spec/v2.0.0.html)

[![license GNU GPLv3](https://img.shields.io/badge/license-GNU%20GPLv3-blue.svg)](https://raw.githubusercontent.com/Dih5/xpecgen/master/LICENSE.txt)
[![DOI](https://zenodo.org/badge/62220331.svg)](https://zenodo.org/badge/latestdoi/62220331)
[![Joss status](http://joss.theoj.org/papers/970f9606afd29308e2dcc77216429ee7/status.svg)](http://joss.theoj.org/papers/970f9606afd29308e2dcc77216429ee7)


[![Build Status](https://travis-ci.org/Dih5/xpecgen.svg?branch=master)](https://travis-ci.org/Dih5/xpecgen)
[![Documentation Status](https://readthedocs.org/projects/xpecgen/badge/?version=latest)](http://xpecgen.readthedocs.io/en/latest/?badge=latest)


A python package with a GUI to calculate **x**-ray s**pec**tra **gen**erated in tungsten anodes, as well as the
bremsstrahlung component in other media.

* [Features](#features)
* [Usage example](#usage-example)
* [Installation](#installation)
* [Documentation](#documentation)
* [Model details](#model-details)
* [Citation](#citation)
* [References](#references)

## Features
* X-ray spectra calculation using the models from [\[1\]](#Ref1).
* HVL calculation.
* Python library and Graphical User Interface.
* Export to xlsx files (Excel Spreadsheet).
* Python API.

## Usage example
### GUI
![alt tag](https://raw.github.com/dih5/xpecgen/master/img/DemoPar.png)
![alt tag](https://raw.github.com/dih5/xpecgen/master/img/DemoPlot.png)
### Python interpreter
![alt tag](https://raw.github.com/dih5/xpecgen/master/img/DemoConsole.png)

## Installation
If you have [pip](https://pip.pypa.io/en/stable/installing/) you can install xpecgen as a package by running
```
pip install xpecgen
```
and then you can launch the GUI by just executing `xpecgen`, see the command line interface by executing `xpecgencli -h`, or check the [demo.py](demo/demo.py) explaining its use as a library,

You will need tk to make use of the GUI. You can check the [advanced guide](advanced.md) if you need help for this.

If you do not have python installed yet, check the [advanced guide](advanced.md).


## Documentation
The updated API documentation is available [here](http://xpecgen.readthedocs.io/en/latest/).

You can also use the python help system to check it:
```python3
from xpecgen import xpecgen as xg
help(xg)
```

## Contributing
See [CONTRIBUTING.md](CONTRIBUTING.md).

## Model details
To have a general overview of the model see [\[1\]](#Ref1).
The bremsstrahlung calculation is done using full interpolations for the electron fluence, so none of the simplifications in section IV.C were used in this implementation. This description of the fluence can be used with others materials, always via the CSDA scaling, if requested. However, note that its accuracy has not been tested.
Both characteristic peaks models in section II.D were implemented. The polynomial one is used by default.
Half-value layers are calculated using the exponential model of attenuation (aka narrow beam geometry). In the GUI they are calculated in the sense of dose, but the library allows for generalizing this to any desired reponse function.

Despite the GUI and the API allow selecting different target materials, note the model only considered tungsten in detail. When a different material is selected, its bremsstrahlung cross-section and range scaling are used instead. However, differences in the electron fluence in the target might affect the results. In particular, the penetration depth is increased in low Z materials in units of the CSDA range, so the results should be used with caution, specially in that case.

## Citation
If you use this application to make use of the models in [\[1\]](#Ref1), you should cite it. If you also want to acknowledge the implementation itself you can also cite [\[2\]](#Ref2).

## References
<a name="Ref1">\[1\]</a> Hernández, G., Fernández F. 2016. "A model of tungsten x-ray anode spectra." Medical Physics, *43* 4655. [doi:10.1118/1.4955120](http://dx.doi.org/10.1118/1.4955120).

<a name="Ref2">\[2\]</a> Hernández, G., Fernández F. 2016. "xpecgen: A program to calculate x-ray spectra generated in tungsten anodes." The Journal of Open Source Software, *00062*. [doi:10.21105/joss.00062](http://dx.doi.org/10.21105/joss.00062).
