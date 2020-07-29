#!/usr/bin/env python
# -*- coding: UTF-8 -*-

#Example of profiling the xpecgen calculations

import cProfile

from xpecgen import xpecgen as xg

cProfile.run('xg.calculate_spectrum(100,12,3,50)')
