#!/usr/bin/env python
# -*- coding: UTF-8 -*-

#xpecgen script example

from __future__ import print_function

from xpecgen import xpecgen as xg

#Define some parameters
E0=100
theta=12
 
#Calculate a spectrum
s=xg.calculate_spectrum(E0,theta,3,150,epsrel=0.5)

#The spectrum can be cloned so the original needn't be recalculated
s2=s.clone()
s2.attenuate(0.12,xg.get_mu(13)) #1.2 mm of Al
s2.attenuate(100,xg.get_mu("air")) #100 cm of Air

#Get some functions to study the spectrum
fluence_to_dose=xg.get_fluence_to_dose()
mu_Al=xg.get_mu(13)
mu_Cu=xg.get_mu(29)

#Normalize the spectrum in the sense of dose
s2.set_norm(value=1,weight=fluence_to_dose)
#Get the norms
print("Number of photons:",s2.get_norm())
print("Energy:",s2.get_norm(lambda x:x),"keV")
print("Dose:",s2.get_norm(fluence_to_dose),"Gy")

#Calculate some HVL
hvl_Al=s2.hvl(0.5,fluence_to_dose,mu_Al)
qvl_Al=s2.hvl(0.25,fluence_to_dose,mu_Al)
print("HVL Al:",10*hvl_Al,"mm")
print("2HVL Al:",10*(qvl_Al-hvl_Al),"mm")
hvl_Cu=s2.hvl(0.5,fluence_to_dose,mu_Cu)
qvl_Cu=s2.hvl(0.25,fluence_to_dose,mu_Cu)
print("HVL Cu:",10*hvl_Cu,"mm")
print("2HVL Cu:",10*(qvl_Cu-hvl_Cu),"mm")

#Export the spectrum as an Excel document
s2.export_xlsx(str(E0)+"keV.xlsx")

#Interact with the spectrum using matplotlib
s2.show_plot()
