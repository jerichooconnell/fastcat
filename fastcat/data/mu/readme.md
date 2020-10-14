# Mu data
This folder contains files describing attenuation coefficients.

**Origin:** the NIST tabulations by [Hubbell & Berger](https://www.nist.gov/pml/x-ray-mass-attenuation-coefficients).

**How obtained:** The tabulation done was fetched with the [physdata](https://github.com/Dih5/physdata) Python package.


**File list:**
- [Z.csv](Z.csv)
- [get_mu.py](get_mu.py)

## Z.csv
- First row: energy in keV
- Second row: attenuation coefficient in cm^-1

## get_mu.py
The Python script used to get this data.
