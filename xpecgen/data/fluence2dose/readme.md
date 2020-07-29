#Fluence to dose data
This folder contains files describing fluence to dose conversion.

**Origin:** H. E. Johns and J. R. Cunningham, *The Physics of Radiology*.

**How obtained:** This tabulation is derived from the one included in [TASMICS](http://dx.doi.org/10.1118/1.4866216). Units were changed and the multiplicative factor is the inverse of the one found there.

Note that the dose depends on the fluence and just not on the spectrum, that's why units of area appear in the conversion factor.

One can also think of the magnitude calculated with this factor as an absolute dose in Gy and understand the photon spectrum is a fluence distribution given in 1/keV/cm^2.


**File list:**
- [f2d.csv](f2d.csv)

## f2d.csv
- First row: energy in keV
- Second row: conversion factor in Gy * cm^2.
