#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
Fastcat a rapid and accurate CBCT simulator.

Contains code from xpecgen.py to simulate spectra:
A module to calculate x-ray spectra generated in tungsten anodes.
"""

from __future__ import print_function

import csv
import logging
import math
import os
import warnings
from bisect import bisect_left
from builtins import map
from glob import glob

import numpy as np
from scipy import integrate, interpolate, optimize

try:
    import matplotlib.pyplot as plt

    plt.ion()
    PLOT_AVAILABLE = True
except ImportError:
    warnings.warn("Unable to import matplotlib. Plotting will be disabled.")
    PLOT_AVAILABLE = False

data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

__author__ = 'JOC'
__version__ = "0.0.1"


def log_interp_1d(xx, yy, kind="linear"):
    """
    Perform interpolation in log-log scale.

    Args:
        xx (List[float]): x-coordinates of the points.
        yy (List[float]): y-coordinates of the points.
        kind (str or int, optional):
            The kind of interpolation in the log-log domain. This is passed to
            scipy.interpolate.interp1d.

    Returns:
        A function whose call method uses interpolation
        in log-log scale to find the value at a given point.
    """
    log_x = np.log(xx)
    log_y = np.log(yy)
    # No big difference in efficiency was found when replacing interp1d by
    # UnivariateSpline
    lin_interp = interpolate.interp1d(log_x, log_y, kind=kind)
    return lambda zz: np.exp(lin_interp(np.log(zz)))


# This custom implementation of dblquad is based in the one in numpy
# (Cf. https://github.com/scipy/scipy
# /blob/v0.16.1/scipy/integrate/quadpack.py#L449 )
# It was modified to work only in rectangular regions (no g(x) nor h(x))
# to set the inner integral epsrel
# and to increase the limit of points taken
def _infunc(x_val, func, c, d, more_args, epsrel=1.49e-8):
    myargs = (x_val,) + more_args
    return integrate.quad(func, c, d, args=myargs, epsrel=epsrel, limit=2000)[
        0
    ]


def custom_dblquad(
    func,
    a,
    b,
    c,
    d,
    args=(),
    epsabs=1.49e-8,
    epsrel=1.49e-8,
    maxp1=50,
    limit=2000,
):
    """
    A wrapper around numpy's dblquad to restrict
    it to a rectangular region and to pass arguments to the 'inner'
    integral.

    Args:
        func: The integrand function f(y,x).
        a (float):
        The lower bound of the second argument in the integrand function.
        b (float):
        The upper bound of the second argument in the integrand function.
        c (float):
        The lower bound of the first argument in the integrand function.
        d (float):
        The upper bound of the first argument in the integrand function.
        args (sequence, optional):
        extra arguments to pass to func.
        epsabs (float, optional):
        Absolute tolerance passed directly to the
        inner 1-D quadrature integration.
        epsrel (float, optional):
        Relative tolerance of the inner 1-D integrals. Default is 1.49e-8.
        maxp1 (float or int, optional):
        An upper bound on the number of Chebyshev moments.
        limit (int, optional):
        Upper bound on the number of cycles (>=3)
        for use with a sinusoidal weighting and an
        infinite end-point.

    Returns:
        (tuple): tuple containing:

            y (float): The resultant integral.

            abserr (float): An estimate of the error.

    """
    return integrate.quad(
        _infunc,
        a,
        b,
        (func, c, d, args, epsrel),
        epsabs=epsabs,
        epsrel=epsrel,
        maxp1=maxp1,
        limit=limit,
    )


def triangle(x, loc=0, size=0.5, area=1):
    """
    The triangle window function centered in loc,
    of given size and area, evaluated at a point.

    Args:
        x: The point where the function is evaluated.
        loc: The position of the peak.
        size: The total.
        area: The area below the function.

    Returns:
        The value of the function.

    """
    # t=abs((x-loc)/size)
    # return 0 if t>1 else (1-t)*abs(area/size)
    return (
        0
        if abs((x - loc) / size) > 1
        else (1 - abs((x - loc) / size)) * abs(area / size)
    )


class Spectrum:
    """
    Set of 2D points and discrete components representing a spectrum.

    A Spectrum can be multiplied by a scalar
    (int, float...) to increase its counts in such a factor.
    Two spectra can be added if they share their x axes and
    their discrete component positions.

    Note: When two spectrum are added it is not checked it
    that addition makes sense. It is the user's responsibility to
    do so.

    Attributes:
        x (:obj:`numpy.ndarray`):
        x coordinates (energy) describing the continuum part of the spectrum.
        y (:obj:`numpy.ndarray`):
        y coordinates (pdf) describing the continuum part of the spectrum.
        discrete (List[List[float]]):
        discrete components of the spectrum, each of the form
        [x, num, rel_x] where:

            * x is the mean position of the peak.
            * num is the number of particles in the peak.
            * rel_x is a characteristic distance where it should extend.
              The exact meaning depends on the windows function.

    """

    def __init__(self):
        """
        Create an empty spectrum.
        """
        self.x = []
        self.y = []
        self.discrete = []

    def clone(self):
        """
        Return a new Spectrum object cloning itself

        Returns:
            :obj:`Spectrum`: The new Spectrum.

        """
        s = Spectrum()
        s.x = list(self.x)
        s.y = self.y[:]
        s.discrete = []
        for a in self.discrete:
            s.discrete.append(a[:])
        return s

    def get_continuous_function(self):
        """
        Get a function representing the continuous part of the spectrum.

        Returns:
            An interpolation function representing the
             continuous part of the spectrum.

        """
        return interpolate.interp1d(
            self.x, self.y, bounds_error=False, fill_value=0
        )

    def get_points(self, peak_shape=triangle, num_discrete=10):
        """
        Returns two lists of coordinates x y
        representing the whole spectrum, both
        the continuous and discrete components.
        The mesh is chosen by extending x
        to include details of the discrete peaks.

        Args:
            peak_shape: The window function
            used to calculate the peaks. See :obj:`triangle` for an example.
            num_discrete: Number of points that are added to mesh in each peak.

        Returns:
            (tuple): tuple containing:

                x2 (List[float]):
                The list of x coordinates (energy) in the whole spectrum.

                y2 (List[float]):
                The list of y coordinates (density) in the whole spectrum.

        """
        if peak_shape is None or self.discrete == []:
            return self.x[:], self.y[:]
        # A mesh for each discrete component:
        discrete_mesh = np.concatenate(
            list(
                map(
                    lambda x: np.linspace(
                        x[0] - x[2],
                        x[0] + x[2],
                        num=num_discrete,
                        endpoint=True,
                    ),
                    self.discrete,
                )
            )
        )
        x2 = sorted(np.concatenate((discrete_mesh, self.x)))
        f = self.get_continuous_function()
        peak = np.vectorize(peak_shape)

        def g(x):
            t = 0
            for ll in self.discrete:
                t += peak(x, loc=ll[0], size=ll[2]) * ll[1]
            return t

        y2 = [f(x) + g(x) for x in x2]
        return x2, y2

    def get_plot(
        self, place, show_mesh=True, prepare_format=True, peak_shape=triangle
    ):
        """
        Prepare a plot of the data in the given place

        Args:
            place:
            The class whose method plot is called to
            produce the plot (e.g., matplotlib.pyplot).
            show_mesh (bool):
            Whether to plot the points over the continuous
            line as circles.
            prepare_format (bool):
            Whether to include ticks and labels in the plot.
            peak_shape: The window function used
            to plot the peaks. See :obj:`triangle` for an example.

        """
        if prepare_format:
            place.tick_params(axis="both", which="major", labelsize=10)
            place.tick_params(axis="both", which="minor", labelsize=8)
            place.set_xlabel("E", fontsize=10, fontweight="bold")
            place.set_ylabel("f(E)", fontsize=10, fontweight="bold")

        x2, y2 = self.get_points(peak_shape=peak_shape)
        if show_mesh:
            place.plot(self.x, self.y, "bo", x2, y2, "b-")
        else:
            place.plot(x2, y2, "b-")

    def show_plot(self, show_mesh=True, block=True):
        """
        Prepare the plot of the data and show it in matplotlib window.

        Args:
            show_mesh (bool):
            Whether to plot the points over the continuous line as circles.
            block (bool): Whether the plot is blocking or non blocking.

        """
        if PLOT_AVAILABLE:
            plt.clf()
            self.get_plot(plt, show_mesh=show_mesh, prepare_format=False)
            plt.xlabel("E")
            plt.ylabel("f(E)")
            plt.gcf().canvas.set_window_title(
                "".join(("fastcat v", __version__))
            )
            plt.show(block=block)
        else:
            warnings.warn(
                "Asked for a plot but matplotlib could not be imported."
            )

    def export_csv(self, route="a.csv", peak_shape=triangle, transpose=False):
        """
        Export the data to a csv file (comma-separated values).

        Args:
            route (str): The route where the file will be saved.
            peak_shape:
            The window function used to plot the peaks.
            See :obj:`triangle` for an example.
            transpose (bool): True to write in two columns, False in two rows.

        """
        x2, y2 = self.get_points(peak_shape=peak_shape)
        with open(route, "w") as csvfile:
            w = csv.writer(csvfile, dialect="excel")
            if transpose:
                w.writerows([list(a) for a in zip(*[x2, y2])])
            else:
                w.writerow(x2)
                w.writerow(y2)

    def get_norm(self, weight=None):
        """
        Return the norm of the spectrum using a weighting function.

        Args:
            weight:
            A function used as a weight to calculate the norm.
            Typical examples are:

                * weight(E)=1 [Photon number]
                * weight(E)=E [Energy]
                * weight(E)=fluence2Dose(E) [Dose]

        Returns:
            (float): The calculated norm.

        """
        if weight is None:
            w = 1
            # def w(x):
            # return 1

        else:
            w = weight
        y2 = list(map(lambda x, y: w * y, self.x, self.y))

        return integrate.simps(y2, x=self.x) + sum(
            [w(a[0]) * a[1] for a in self.discrete]
        )

    def set_norm(self, value=1, weight=None):
        """
        Set the norm of the spectrum using a weighting function.

        Args:
            value (float):
            The norm of the modified spectrum in the given convention.
            weight: A function used as a weight to calculate the norm.
            Typical examples are:

                * weight(E)=1 [Photon number]
                * weight(E)=E [Energy]
                * weight(E)=fluence2Dose(E) [Dose]
        """
        norm = self.get_norm(weight=weight) / value
        self.y = [a / norm for a in self.y]
        self.discrete = [[a[0], a[1] / norm, a[2]] for a in self.discrete]

    def hvl(self, value=0.5, weight=lambda x: 1, mu=lambda x: 1, energy_min=0):
        """
        Calculate a generalized half-value-layer.

        This method calculates the depth of a material needed
        for a certain dosimetric magnitude to decrease in a given factor.

        Args:
            value (float): The factor the desired
            magnitude is decreased. Must be in [0, 1].
            weight: A function used as a weight
            to calculate the norm. Typical examples are:

                * weight(E)=1 [Photon number]
                * weight(E)=E [Energy]
                * weight(E)=fluence2Dose(E) [Dose]
            mu: The energy absorption coefficient as a function of energy.
            energy_min (float): A low-energy cutoff to use in the calculation.

        Returns:
            (float): The generalized hvl in cm.
        """
        # cutoff. However, such a high cutoff
        # would probably make no sense

        try:
            # Use low-energy cutoff
            low_index = bisect_left(self.x, energy_min)
            x = self.x[low_index:]
            y = self.y[low_index:]
            # Normalize to 1 with weighting function
            y2 = list(map(lambda a, b: weight(a) * b, x, y))
            discrete2 = [weight(a[0]) * a[1] for a in self.discrete]
            n2 = integrate.simps(y2, x=x) + sum(discrete2)
            y3 = [a / n2 for a in y2]
            discrete3 = [
                [a[0], weight(a[0]) * a[1] / n2] for a in self.discrete
            ]
            # Now we only need to add attenuation as a function of depth
            f = (
                lambda t: integrate.simps(
                    list(map(lambda a, b: b * math.exp(-mu(a) * t), x, y3)),
                    x=x,
                )
                + sum([c[1] * math.exp(-mu(c[0]) * t) for c in discrete3])
                - value
            )
            # Search the order of magnitude of
            # the root (using the fact that f is
            # monotonically decreasing)
            a = 1.0
            if f(a) > 0:
                while f(a) > 0:
                    a *= 10.0
                # Now f(a)<=0 and f(a*0.1)>0
                return optimize.brentq(f, a * 0.1, a)
            while f(a) < 0:
                a *= 0.1
            # Now f(a)>=0 and f(a*10)<0
            return optimize.brentq(f, a, a * 10.0)

        except ValueError:
            warnings.warn("Interpolation boundary error")
            return 0

    def attenuate(self, depth=1, mu=lambda x: 1):
        """
        Attenuate the spectrum as if it passed
        thorough a given depth of material with attenuation
        described by a given
        attenuation coefficient. Consistent units should be used.

        Args:
            depth: The amount of material (typically in cm).
            mu: The energy-dependent absorption coefficient
            (typically in cm^-1).
        """

        self.y = list(
            map(lambda x, y: y * math.exp(-mu(x) * depth), self.x, self.y)
        )
        self.discrete = list(
            map(
                lambda l: [l[0], l[1] * math.exp(-mu(l[0]) * depth), l[2]],
                self.discrete,
            )
        )

    def load(self, spectrum_file):
        '''
        Load spectrum_file from file. Loads one of the text files found in the
        data/MV_spectra directory do not add file extension to spectrum_file
        '''

        energies = []
        fluence = []

        with open(
            os.path.join(data_path, "MV_spectra", f"{spectrum_file}.txt")
        ) as f:
            for line in f:
                energies.append(float(line.split()[0]))
                fluence.append(float(line.split()[1]))

        # Check if MV

        self.x = np.array(energies) * 1000  # to keV
        self.y = np.array(fluence)

    def __add__(self, other):
        """Add two instances, assuming that makes sense."""
        if not isinstance(
            other, Spectrum
        ):  # so s+0=s and sum([s1, s2,...]) makes sense
            return self
        s = Spectrum()
        s.x = self.x
        s.y = [a + b for a, b in zip(self.y, other.y)]
        s.discrete = [
            [a[0], a[1] + b[1], a[2]]
            for a, b in zip(self.discrete, other.discrete)
        ]
        return s

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        """Multiply the counts by an scalar."""
        s2 = self.clone()
        s2.y = [a * other for a in self.y]
        s2.discrete = [[a[0], a[1] * other, a[2]] for a in self.discrete]
        return s2

    def __rmul__(self, other):
        return self.__mul__(other)


def get_fluence(e_0=100.0):
    """
    Returns a function representing the electron
    fluence with the distance in CSDA units.

    Args:
        e_0 (float): The kinetic energy whose
        CSDA range is used to scale the distances.

    Returns:
        A function representing fluence(x,u) with x in CSDA units.
    """
    # List of available energies
    e0_str_list = list(
        map(
            lambda x: (os.path.split(x)[1]).split(".csv")[0],
            glob(os.path.join(data_path, "fluence", "*.csv")),
        )
    )
    e0_list = sorted(list(map(int, list(filter(str.isdigit, e0_str_list)))))

    e_closest = min(e0_list, key=lambda x: abs(x - e_0))

    with open(os.path.join(data_path, "fluence/grid.csv"), "r") as csvfile:
        r = csv.reader(
            csvfile, delimiter=" ", quotechar="|", quoting=csv.QUOTE_MINIMAL
        )
        t = next(r)
        x = np.array([float(a) for a in t[0].split(",")])
        t = next(r)
        u = np.array([float(a) for a in t[0].split(",")])
    t = []
    with open(
        os.path.join(data_path, "fluence", "".join([str(e_closest), ".csv"])),
        "r",
    ) as csvfile:
        r = csv.reader(
            csvfile, delimiter=" ", quotechar="|", quoting=csv.QUOTE_MINIMAL
        )
        for row in r:
            t.append([float(a) for a in row[0].split(",")])
    t = np.array(t)
    f = interpolate.RectBivariateSpline(x, u, t, kx=1, ky=1)
    # Note f is returning numpy 1x1 arrays
    return f
    # return lambda x,u:f(x,u)[0]


def get_cs(e_0=100, z=74):
    """
    Returns a function representing the scaled bremsstrahlung cross_section.

    Args:
        e_0 (float): The electron kinetic energy, used to scale u=e_e/e_0.
        z (int): Atomic number of the material.

    Returns:
        A function representing cross_section(e_g,u)
        in mb/keV, with e_g in keV.
    """
    # NOTE: Data is given for E0>1keV. CS values below this
    # level should be used with caution.
    # The default behaviour is to keep it constant
    with open(os.path.join(data_path, "cs/grid.csv"), "r") as csvfile:
        r = csv.reader(
            csvfile, delimiter=" ", quotechar="|", quoting=csv.QUOTE_MINIMAL
        )
        t = next(r)
        e_e = np.array([float(a) for a in t[0].split(",")])
        log_e_e = np.log10(e_e)
        t = next(r)
        k = np.array([float(a) for a in t[0].split(",")])
    t = []
    with open(os.path.join(data_path, "cs/%d.csv" % z), "r") as csvfile:
        r = csv.reader(
            csvfile, delimiter=" ", quotechar="|", quoting=csv.QUOTE_MINIMAL
        )
        for row in r:
            t.append([float(a) for a in row[0].split(",")])
    t = np.array(t)
    scaled = interpolate.RectBivariateSpline(log_e_e, k, t, kx=3, ky=1)
    m_electron = 511
    z2 = z * z
    return (
        lambda e_g, u: (u * e_0 + m_electron) ** 2
        * z2
        / (u * e_0 * e_g * (u * e_0 + 2 * m_electron))
        * (scaled(np.log10(u * e_0), e_g / (u * e_0)))
    )


def get_mu(z=74):
    """
    Returns a function representing an energy-dependent
    attenuation coefficient.

    Args:
        z (int or str): The identifier of the material in
        the data folder, typically the atomic number.

    Returns:
        The attenuation coefficient mu(E) in cm^-1 as
        a function of the energy measured in keV.
    """
    with open(
        os.path.join(data_path, "mu", "".join([str(z), ".csv"])), "r"
    ) as csvfile:
        r = csv.reader(
            csvfile, delimiter=" ", quotechar="|", quoting=csv.QUOTE_MINIMAL
        )
        t = next(r)
        x = [float(a) for a in t[0].split(",")]
        t = next(r)
        y = [float(a) for a in t[0].split(",")]
    return log_interp_1d(x, y)


def get_csda(z=74):
    """
    Returns a function representing the CSDA range in tungsten.

    Args:
        z (int): Atomic number of the material.

    Returns:
        The CSDA range in cm in tungsten as a
        function of the electron kinetic energy in keV.
    """
    with open(os.path.join(data_path, "csda/%d.csv" % z), "r") as csvfile:
        r = csv.reader(
            csvfile, delimiter=" ", quotechar="|", quoting=csv.QUOTE_MINIMAL
        )
        t = next(r)
        x = [float(a) for a in t[0].split(",")]
        t = next(r)
        y = [float(a) for a in t[0].split(",")]
    return interpolate.interp1d(x, y, kind="linear")


def get_mu_csda(e_0, z=74):
    """
    Returns a function representing the CSDA-scaled
    energy-dependent attenuation coefficient in tungsten.

    Args:
        e_0 (float): The electron initial kinetic energy.
        z (int): Atomic number of the material.

    Returns:
        The attenuation coefficient mu(E) in CSDA units
        as a function of the energy measured in keV.
    """
    mu = get_mu(z)
    csda = get_csda(z=z)(e_0)
    return lambda e: mu(e) * csda


def get_fluence_to_dose():
    """
    Returns a function representing the weighting
    factor which converts fluence to dose.

    Returns:
        A function representing the weighting factor
        which converts fluence to dose in Gy * cm^2.
    """
    with open(os.path.join(data_path, "fluence2dose/f2d.csv"), "r") as csvfile:
        r = csv.reader(
            csvfile, delimiter=" ", quotechar="|", quoting=csv.QUOTE_MINIMAL
        )
        t = next(r)
        x = [float(a) for a in t[0].split(",")]
        t = next(r)
        y = [float(a) for a in t[0].split(",")]
    return interpolate.interp1d(x, y, kind="linear")


def get_source_function(fluence, cs, mu, theta, e_g, phi=0.0):
    """
    Returns the attenuated source function
    (Eq. 2 in the paper) for the given parameters.

    An E_0-dependent factor (the fraction found there) is excluded.
    However, the E_0 dependence is removed in
    integrate_source.

    Args:
        fluence: The function representing the fluence.
        cs: The function representing the bremsstrahlung cross-section.
        mu: The function representing the attenuation coefficient.
        theta (float): The emission angle in degrees,
        the anode's normal being at 90º.
        e_g (float): The emitted photon energy in keV.
        phi (float): The elevation angle in degrees,
        the anode's normal being at 12º.

    Returns:
        The attenuated source function s(u,x).
    """
    factor = (
        -mu(e_g) / math.sin(math.radians(theta)) / math.cos(math.radians(phi))
    )
    return lambda u, x: fluence(x, u) * cs(e_g, u) * math.exp(factor * x)


def integrate_source(
    fluence,
    cs,
    mu,
    theta,
    e_g,
    e_0,
    phi=0.0,
    x_min=0.0,
    x_max=0.6,
    epsrel=0.1,
    z=74,
):
    """
    Find the integral of the attenuated source function.

    An E_0-independent factor is excluded
    (i.e., the E_0 dependence on get_source_function
    is taken into account here).

    Args:
        fluence: The function representing the fluence.
        cs: The function representing the bremsstrahlung cross-section.
        mu: The function representing the attenuation coefficient.
        theta (float): The emission angle in degrees,
        the anode's normal being at 90º.
        e_g: (float): The emitted photon energy in keV.
        e_0 (float): The electron initial kinetic energy.
        phi (float): The elevation angle in degrees,
        the anode's normal being at 12º.
        x_min: The lower-bound of the integral in depth,
        scaled by the CSDA range.
        x_max: The upper-bound of the integral in depth,
        scaled by the CSDA range.
        epsrel: The relative tolerance of the integral.
        z (int): Atomic number of the material.

    Returns:
        float: The value of the integral.
    """
    if e_g >= e_0:
        return 0
    f = get_source_function(fluence, cs, mu, theta, e_g, phi=phi)
    (y, y_err) = custom_dblquad(
        f, x_min, x_max, e_g / e_0, 1, epsrel=epsrel, limit=100
    )
    # The factor includes n_med, its units being 1/(mb * r_CSDA).
    # We only take into account the r_CSDA dependence.
    y *= get_csda(z=z)(e_0)
    return y


def add_char_radiation(s, method="fraction_above_poly"):
    """
    Adds characteristic radiation to a calculated bremsstrahlung spectrum,
    assuming it is a tungsten-generated spectrum

    If a discrete component already exists in the spectrum, it is replaced.

    Args:
        s (:obj:`Spectrum`): The spectrum whose discrete
        component is recalculated.
        method (str): The method to use to calculate the
        discrete component. Available methods include:

            * 'fraction_above_linear': Use a linear relation
            between bremsstrahlung above the K-edge and peaks.
            * 'fraction_above_poly': Use polynomial fits between
            bremsstrahlung above the K-edge and peaks.

    """
    s.discrete = []
    if s.x[-1] < 69.51:  # If under k edge, no char radiation
        return

    f = s.get_continuous_function()
    norm = integrate.quad(f, s.x[0], s.x[-1], limit=2000)[0]
    fraction_above = integrate.quad(f, 74, s.x[-1], limit=2000)[0] / norm

    if method == "fraction_above_linear":
        s.discrete.append([58.65, 0.1639 * fraction_above * norm, 1])
        s.discrete.append([67.244, 0.03628 * fraction_above * norm, 1])
        s.discrete.append([69.067, 0.01410 * fraction_above * norm, 1])
    else:
        if method != "fraction_above_poly":
            logging.info(
                "WARNING: Unknown char radiation calculation method."
                "Using fraction_above_poly"
            )
        s.discrete.append(
            [
                58.65,
                (
                    0.1912 * fraction_above
                    - 0.00615 * fraction_above ** 2
                    - 0.1279 * fraction_above ** 3
                )
                * norm,
                1,
            ]
        )
        s.discrete.append(
            [
                67.244,
                (
                    0.04239 * fraction_above
                    + 0.002003 * fraction_above ** 2
                    - 0.02356 * fraction_above ** 3
                )
                * norm,
                1,
            ]
        )
        s.discrete.append(
            [
                69.067,
                (
                    0.01437 * fraction_above
                    + 0.002346 * fraction_above ** 2
                    - 0.009332 * fraction_above ** 3
                )
                * norm,
                1,
            ]
        )

    return


def console_monitor(a, b):
    """
    Simple monitor function which can be used with :obj:`calculate_spectrum`.

    Prints in stdout 'a/b'.

    Args:
        a: An object representing the completed amount
        (e.g., a number representing a part...).
        b: An object representing the total amount
        (... of a number representing a total).
    """
    print("Calculation: ", a, "/", b)


def calculate_spectrum_mesh(
    e_0, theta, mesh, phi=0.0, epsrel=0.2, monitor=console_monitor, z=74
):
    """
    Calculates the x-ray spectrum for given parameters.
    Characteristic peaks are also calculated by add
    char_radiation, which is called with the default parameters.

    Args:
        e_0 (float): Electron kinetic energy in keV
        theta (float): X-ray emission angle in degrees, the normal being at 90º
        mesh (list of float or ndarray): The photon
        energies where the integral will be evaluated
        phi (float): X-ray emission elevation angle in degrees.
        epsrel (float): The tolerance parameter used in numeric integration.
        monitor: A function to be called after
        each iteration with arguments finished_count,
        total_count. See for example :obj:`console_monitor`.
        z (int): Atomic number of the material.

    Returns:
        :obj:`Spectrum`: The calculated spectrum
    """
    # Prepare spectrum
    s = Spectrum()
    s.x = mesh
    mesh_len = len(mesh)
    # Prepare integrand function
    fluence = get_fluence(e_0)
    cs = get_cs(e_0, z=z)
    mu = get_mu_csda(e_0, z=z)

    # quad may raise warnings about the numerical integration method,
    # which are related to the estimated accuracy. Since this is not relevant,
    # they are suppressed.
    warnings.simplefilter("ignore")

    for i, e_g in enumerate(s.x):
        s.y.append(
            integrate_source(
                fluence, cs, mu, theta, e_g, e_0, phi=phi, epsrel=epsrel, z=z
            )
        )
        if monitor is not None:
            monitor(i + 1, mesh_len)

    if z == 74:
        add_char_radiation(s)

    return s


def calculate_spectrum(
    e_0,
    theta,
    e_min,
    num_e,
    phi=0.0,
    epsrel=0.2,
    monitor=console_monitor,
    z=74,
):
    """
    Calculates the x-ray spectrum for given parameters.
    Characteristic peaks are also calculated
    by add_char_radiation, which is called with the default parameters.

    Args:
        e_0 (float): Electron kinetic energy in keV
        theta (float): X-ray emission angle in degrees, the normal being at 90º
        e_min (float): Minimum kinetic energy
        to calculate in the spectrum in keV
        num_e (int): Number of points to calculate in the spectrum
        phi (float): X-ray emission elevation angle in degrees.
        epsrel (float): The tolerance parameter used in numeric integration.
        monitor: A function to be called after
        each iteration with arguments finished_count, total_count.
        See for example :obj:`console_monitor`.
        z (int): Atomic number of the material.

    Returns:
        :obj:`Spectrum`: The calculated spectrum

    """
    return calculate_spectrum_mesh(
        e_0,
        theta,
        np.linspace(e_min, e_0, num=num_e, endpoint=True),
        phi=phi,
        epsrel=epsrel,
        monitor=monitor,
        z=z,
    )
