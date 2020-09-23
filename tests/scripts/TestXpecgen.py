#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TestXpecgen.py: Tests for  the `xpecgen` package.
"""

import unittest
import math

from xpecgen import xpecgen as xg

s = xg.calculate_spectrum(100, 12, 3, 10, epsrel=0.5, monitor=None)


class XpecgenTest(unittest.TestCase):
    def test_clone(self):
        """Test the Spectrum clone method"""
        s2 = s.clone()
        # Check same values
        self.assertEqual(list(s.x), list(s2.x))
        self.assertEqual(list(s.y), list(s2.y))
        self.assertEqual(list(s.discrete), list(s2.discrete))
        # Check alteration does not alter both instances
        s.x[0] = 10.0
        s.y[0] = 10.0
        s.discrete[0][0] = 10.0
        self.assertNotEqual(s.x[0], s2.x[0])
        self.assertNotEqual(s.y[0], s2.y[0])
        self.assertNotEqual(s.discrete[0][0], s2.discrete[0][0])

    def test_get_continuous_function(self):
        """Test the get_continuous_function method"""
        f = s.get_continuous_function()
        for p in zip(s.x, s.y):
            self.assertAlmostEqual(p[1], f(p[0]))

    def test_norm_functions(self):
        """Test the get_norm and set_norm methods"""
        s2 = s.clone()
        for w in [lambda x: 1, lambda x: x, xg.get_fluence_to_dose()]:
            s2.set_norm(13.0, w)
            self.assertAlmostEqual(s2.get_norm(w), 13.0)

    def test_attenuate(self):
        """Test the attenuate method"""
        # Unit distance, unit energy-independent attenuation coefficient -> attenuation with a factor 1/e
        s2 = s.clone()
        s2.attenuate(1, lambda x: 1)
        # Check continuous component changed
        for p in zip(s.y, s2.y):
            self.assertAlmostEqual(p[0] / math.e, p[1])
        # Check discrete component change
        for p in zip(s.discrete, s2.discrete):
            self.assertAlmostEqual(p[0][1] / math.e, p[1][1])

    def test_HVL_values(self):
        """Test to reproduce the values in Table III of Med. Phys. 43, 4655 (2016)"""

        # Calculate the emission spectra
        num_div = 20  # Points in each spectrum (quick calculation)
        s50 = xg.calculate_spectrum(50, 12, 3, num_div, epsrel=0.5, monitor=None)
        s80 = xg.calculate_spectrum(80, 12, 3, num_div, epsrel=0.5, monitor=None)
        s100 = xg.calculate_spectrum(100, 12, 3, num_div, epsrel=0.5, monitor=None)

        # Attenuate them
        s50.attenuate(0.12, xg.get_mu(13))  # 1.2 mm of Al
        s50.attenuate(100, xg.get_mu("air"))  # 100 cm of Air
        s80.attenuate(0.12, xg.get_mu(13))  # 1.2 mm of Al
        s80.attenuate(100, xg.get_mu("air"))  # 100 cm of Air
        s100.attenuate(0.12, xg.get_mu(13))  # 1.2 mm of Al
        s100.attenuate(100, xg.get_mu("air"))  # 100 cm of Air

        # Functions to calculate HVL in the sense of dose in Al
        fluence_to_dose = xg.get_fluence_to_dose()
        mu_al = xg.get_mu(13)

        # HVL in mm
        hvl50 = 10 * s50.hvl(0.5, fluence_to_dose, mu_al)
        hvl80 = 10 * s80.hvl(0.5, fluence_to_dose, mu_al)
        hvl100 = 10 * s100.hvl(0.5, fluence_to_dose, mu_al)

        self.assertAlmostEqual(hvl100, 2.37, places=1)
        self.assertAlmostEqual(hvl80, 1.85, places=1)
        self.assertAlmostEqual(hvl50, 1.20, places=1)

    def test_linear_combination(self):
        """Test linear combinations with Spectrum instances"""
        s2 = s.clone()
        s3 = s2+s2
        s4 = s2*2
        # Check same values
        self.assertEqual(list(s3.x), list(s4.x))
        self.assertEqual(list(s3.y), list(s4.y))
        self.assertEqual(list(s3.discrete), list(s4.discrete))

if __name__ == "__main__":
    unittest.main()
