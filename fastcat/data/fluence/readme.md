# Fluence data
This folder contains files describing electron penetration in tungsten.

**Origin:** the set of FLUKA simulations explained in the [model paper](http://dx.doi.org/10.1118/1.4955120).

**How obtained:** By the numerical interpolation described in that reference.

**File list:**
- [Grid.csv](grid.csv)
- [E0.csv](100.csv)


## Grid.csv
- First line: Depth in units of the csda range (x)
- Second line: Fraction of kinetic energy retained by the electron (u)

## E0.csv
Matrix of differential fluence in arbitrary units depending on:
- Row: x (units of the csda range)
- Column: u

