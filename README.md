# Fastcat

![anim](anim.gif)

A rapid, highly-realistic, scanner specific CBCT simulator.

* [Features](#features)
* [Installation](#installation)

## Features
* X-ray spectra calculation using the models from xpecgen
* Detector response based on MC optical simulation of Scintillators
* High resolution Catphan phantoms
* Bowtie filters, flattening filters, ant-scatter grid models
* Python library and Graphical User Interface

## Installation

This package is not currently released. A release will be made after supporting papers are published.

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

<a name="Ref2">\[2\]</a> Hernández, G., Fernández F. 2016. "xpecgen: A program to calculate x-ray spectra generated in tungsten anodes." The Journal of Open Source Software, *00062*. [doi:10.21105/joss.00062](http://dx.doi.org/10.21105/joss.00062). -->
