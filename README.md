# Fastcat

[![PyPI Version][pypi-image]][pypi-url]
[![Build Status][build-image]][build-url]

![anim](anim.gif)

> A rapid, highly-realistic, scanner specific CBCT simulator.

* [Features](#features)
* [Installation](#installation)

## Features
* X-ray spectra calculation using the models from xpecgen
* Detector response based on MC optical simulation of Scintillators
* High resolution Catphan phantoms
* Bowtie filters, flattening filters, ant-scatter grid models
* Python library and Graphical User Interface

## Installation

Fastcat requires a few dependencies to run. Most importantly you need a cuda capable GPU.

### Dependencies

* CUDA
* TIGRE
* Scientific python installation (anaconda ect.)

Fastcat requires TIGRE which can be installed using directions here:

https://github.com/CERN/TIGRE/blob/master/Frontispiece/python_installation.md

After installing TIGRE I recommend cloning the repository and installing it in developer mode so that you can access the files

`git clone https://github.com/jerichooconnell/fastcat`

`cd fastcat && python setup.py develop`

Fastcat is a work in progress and there are quite a few bugs to work out. So feel free to file issues and I'll take a look.

## Citation
If you use this application in a publication please cite the two publications on the development of fastcat so that I can climb rungs in the academic rat race please and thanks. Also have a read if you like, there are preprints on arxiv if you can't get through the paywall:

<a name="Ref1">\[1\]</a> O'Connell, J. and Bazalova-Carter, M. (2021), fastCAT: Fast cone beam CT (CBCT) simulation. Med. Phys., 48: 4448-4458. [doi.org/10.1002/mp.15007](https://doi.org/10.1002/mp.15007).

<a name="Ref2">\[2\]</a>  O’Connell, J, Lindsay, C, Bazalova-Carter, M. Experimental validation of Fastcat kV and MV cone beam CT (CBCT) simulator. Med. Phys. 2021; 48: 6869– 6880. [doi.org/10.1002/mp.15243](https://doi.org/10.1002/mp.15243).

<!-- Badges -->

[pypi-image]: https://img.shields.io/pypi/v/podsearch
[pypi-url]: https://pypi.org/project/podsearch/
[build-image]: https://github.com/nalgeon/podsearch-py/actions/workflows/build.yml/badge.svg
[build-url]: https://github.com/nalgeon/podsearch-py/actions/workflows/build.yml

<!-- ## Documentation
The updated API documentation is available [here](http://xpecgen.readthedocs.io/en/latest/).

You can also use the python help system to check it:
```python3
from xpecgen import xpecgen as xg
help(xg)
``` -->

<!-- ## Contributing
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

 -->
