# Cross-section data
This folder contains files describing bremsstrahlung production cross section.

**Origin:** This data comes from the tabulations by Seltzer&Berger (see [Ref. 1](http://www.sciencedirect.com/science/article/pii/0168583X85907074) and [Ref. 2](http://www.sciencedirect.com/science/article/pii/0092640X86900148))

**How obtained:** The tabulation done here was derived from the one in PENELOPE.

**File list:**
- [Grid.csv](grid.csv).
- [Z.csv](Z.csv).

## Grid.csv
- First line: Electron kinetic energy in keV.
- Second line: Fraction of energy transfered to emmited photon.

## Z.csv
Matrix of scaled differential cross section in mbarn depending on:
- Row: Electron kinetic energy (keV).
- Column: Fraction of energy transfered to the photon.
Each of them refering to the grid in [grid.csv](grid.csv).

A scaled cross-section is defined as beta^2/Z^2 \* E_gamma \* dsigma/dE_gamma.
