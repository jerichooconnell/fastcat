#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# xpecgen electron distribution example

from __future__ import print_function

import numpy as np
from scipy.stats import gamma

from xpecgen import xpecgen as xg


# Cumulative density function of a Maxwell-Boltzmann distribution
def mb_cdf(x, kT, alpha=1.5):
    return gamma.cdf(x, alpha, scale=kT)


theta = 12
kT = 60
# Energies where the spectrum will be calculated
gamma_mesh = np.linspace(3, 100, num=30)
# Points where the function will be evaluated
electron_dist_x = np.linspace(10, 120, num=30)

# You will probably want to increase the sampling points in a real calculation.

# Each will represent an interval whose density is given by the CDF. The additional endpoints are 0 and inf.
mesh_points = np.concatenate(([0], (electron_dist_x[1:] + electron_dist_x[:-1]) / 2, [np.inf]))

electron_dist_y = [mb_cdf(e_1, kT) - mb_cdf(e_0, kT) for e_1, e_0 in zip(mesh_points[1:], mesh_points[:-1])]

# Calculate the spectra for each energy
s_list = [xg.calculate_spectrum_mesh(e_0, theta, mesh=gamma_mesh, epsrel=0.8, monitor=None) for e_0 in electron_dist_x]

# Add them using the density in each interval
s = sum([w * sp for sp, w in zip(s_list, electron_dist_y)])

# Show the result
s.show_plot(block=True)
