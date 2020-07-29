#!/usr/bin/env python
# -*- coding: UTF-8 -*-

#Example of script to reproduce the values in Table III of Med. Phys. 43, 4655 (2016)

from __future__ import print_function

from xpecgen import xpecgen as xg



num_div=100 #Points in each spectrum

#Calculate the emission spectra
s50=xg.calculate_spectrum(50,12,3,num_div,epsrel=0.5)
s80=xg.calculate_spectrum(80,12,3,num_div,epsrel=0.5)
s100=xg.calculate_spectrum(100,12,3,num_div,epsrel=0.5)

#Attenuate them
s50.attenuate(0.12,xg.get_mu(13)) #1.2 mm of Al
s50.attenuate(100,xg.get_mu("air")) #100 cm of Air
s80.attenuate(0.12,xg.get_mu(13)) #1.2 mm of Al
s80.attenuate(100,xg.get_mu("air")) #100 cm of Air
s100.attenuate(0.12,xg.get_mu(13)) #1.2 mm of Al
s100.attenuate(100,xg.get_mu("air")) #100 cm of Air

#Functions to calculate HVL in the sense of dose in Al
fluence_to_dose=xg.get_fluence_to_dose()
mu_Al=xg.get_mu(13)

print("HVL in Al:")
print("100 keV:",10 * s100.hvl(0.5,fluence_to_dose,mu_Al),"mm")
print("80 keV:",10 * s80.hvl(0.5,fluence_to_dose,mu_Al),"mm")
print("50 keV:",10 * s50.hvl(0.5,fluence_to_dose,mu_Al),"mm")
