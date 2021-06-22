#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
Fastcat a rapid and accurate CBCT simulator.

Contains code from xpecgen.py to simulate spectra:
A module to calculate x-ray spectra generated in tungsten anodes.
"""

from __future__ import print_function

import logging
import os
import sys

import numpy as np
import tigre
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit
from scipy.signal import fftconvolve
from scipy.interpolate import interp1d


try:
    import matplotlib.pyplot as plt
    PLOT_AVAILABLE = True
except ImportError:
    PLOT_AVAILABLE = False

from fastcat import spectrum

logging.basicConfig(
    stream=sys.stdout,
    format="[%(asctime)s] {%(filename)s:%"
    "(lineno)d} %(levelname)s - %(message)s",
    level=logging.DEBUG,
)

__author__ = "Jericho OConnell"
__version__ = "0.0.1"

data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

mu_en_water = np.array(
    [
        4.944,
        0.5503,
        0.1557,
        0.06947,
        0.04223,
        0.0319,
        0.027678,
        0.02597,
        0.025434,
        0.02546,
        0.03192,
        0.03299,
        0.032501,
        0.031562,
        0.03103,
        0.02608,
        0.02066,
        0.01806,
    ]
)
mu_water = np.array(
    [
        5.329,
        0.8096,
        0.3756,
        0.2683,
        0.2269,
        0.2059,
        0.19289,
        0.1837,
        0.176564,
        0.1707,
        0.1186,
        0.09687,
        0.083614,
        0.074411,
        0.07072,
        0.04942,
        0.03403,
        0.0277,
    ]
)


class Phantom:
    '''
    Super that contains the simulation method
    '''

    def return_projs(
        self,
        kernel,
        spectra,
        angles,
        nphoton=None,
        mgy=0.0,
        return_dose=False,
        det_on=True,
        scat_on=True,
        tigre_works=True,
        convolve_on=True,
        ASG=False,
        **kwargs,
    ):
        """
        The main simulation, a call to return generates the
        projection from a detector, phantom, spectrum, and angles
        projections are stored in the phantom object

        Input
        -----
        detector: Fastcat detector object
            Determines the energy efficiency
            and optical spread of the simulation
        spectra: Fastcat spectra object
            Energy spectra that is used in the simulation
        angles: 1d np.array or list
            The angles to return projections for in radians
        nphoton: float
            Option to specify the number of photons
            used in the simulation, will modify
            the noise. Defaults to None which means no noise.
        mgy: float
            Option to specify the dose to each projection in mgy,
            will modify the noise.
            Defaults to 0.0 which will give no noise.

        Returns
        -------
        Projections are stored at phantom.projs

        Keyword Arguments
        -----------------
            return_dose: Bool
                This is an option mostly for tests.
                Returns some partial dose calculations.
            det_on: Bool
                This is an option to turn the detector
                off which will give perfect energy
                absorption. Default True (on)
            scat_on: Bool
                This is an option to remove scatter
                from the simulation. Default True (on)
            convolve_on: Bool
                Option to give a perfect response
                for the detector where spatial resolution
                is not degraded by optical spread.
                Default True (on)
            ASG: Bool
                Option to include an anti scatter grid.
                The anti scatter grid has a primary
                transmission factor of 0.72 and a scatter
                transmission factor of 0.32.
                Default False (off)
            filter: string
                String which specifies one of the
                filters in data/filters/ these filter
                the beam before it hits the phantom.

        Example (MV CBCT)
        -------

        import fastcat.fastcat as fc
        import numpy as np
        import matplotlib.pyplot as plt

        s = fc.Spectrum()
        s.load('W_spectrum_6') # 6 MV Tungsten
        det = fc.Detector(s,'CuGOS-784-micrometer')

        angles = np.linspace(0,2*pi,90,endpoint=False)
        phantom = fc.Catphan_404()
        phantom.return_projs(det,s,angles,mgy=0.4)
        phantom.recon('FDK')

        plt.figure() # Show one of the reconstructed images
        plt.imshow(phantom.img[5])
        """

        return_intensity = False

        if "test" in kwargs.keys():
            if kwargs["test"] == 1:
                det_on = False
            if kwargs["test"] == 2:
                return_intensity = True

        if "verbose" in kwargs.keys():
            if kwargs["verbose"] == 0:
                level = logging.WARNING
                logger = logging.getLogger()
                logger.setLevel(level)
                for handler in logger.handlers:
                    handler.setLevel(level)

        bowtie_on = False

        if "bowtie" in kwargs.keys():
            if kwargs["bowtie"]:
                bowtie_on = True
                logging.info(f'Initializing filter {kwargs["filter"]}')

        self.tigre_works = tigre_works
        self.angles = angles

        # ----------------------------------------------------------
        # --- Making the weights for the different energies --------
        # ----------------------------------------------------------

        # These are what I used in the Monte Carlo
        deposition = np.load(
            kernel.deposition_efficiency_file, allow_pickle=True
        )

        # csi has two extra kv energies
        if len(deposition[0]) == 18:
            original_energies_keV = np.array(
                [
                    10,
                    20,
                    30,
                    40,
                    50,
                    60,
                    70,
                    80,
                    90,
                    100,
                    300,
                    500,
                    700,
                    900,
                    1000,
                    2000,
                    4000,
                    6000,
                ]
            )
            mu_en_water2 = mu_en_water
            mu_water2 = mu_water
        else:
            logging.info("This is a small edep file starting at 30")
            original_energies_keV = np.array(
                [
                    30,
                    40,
                    50,
                    60,
                    70,
                    80,
                    90,
                    100,
                    300,
                    500,
                    700,
                    900,
                    1000,
                    2000,
                    4000,
                    6000,
                ]
            )
            mu_en_water2 = mu_en_water[2:]
            mu_water2 = mu_water[2:]

        #         spectra.x, spectra.y = spectra.get_points()

        # Set the last value to zero so that
        # the linear interpolation doesn't mess up
        spectra.y[-1] = 0

        # Loading the file from the monte carlo
        # This is a scaling factor that I found
        # to work to convert energy deposition to photon probability eta
        deposition_summed = (
            deposition[0] / (original_energies_keV / 1000) / 1000000
        )

        # Binning to get the fluence per energy
        large_energies = np.linspace(0, 6000, 3001)
        f_flu = interp1d(
            np.insert(spectra.x, (0, -1), (0, 6000)),
            np.insert(spectra.y, (0, -1), (0, spectra.y[-1])),
        )
        f_dep = interp1d(
            np.insert(original_energies_keV, 0, 0),
            np.insert(deposition_summed, 0, 0),
        )
        f_e = interp1d(
            np.insert(original_energies_keV, 0, 0),
            np.insert((deposition[0]), 0, 0),
        )

        weights_xray_small = np.zeros(len(original_energies_keV))
        weights_energies = np.zeros(len(original_energies_keV))
        fluence_small = np.zeros(len(original_energies_keV))

        if det_on:
            weights_xray = f_flu(large_energies) * f_dep(large_energies)
            weights_e_long = f_flu(large_energies) * f_e(large_energies)
        else:
            weights_xray = f_flu(large_energies)
            weights_e_long = f_flu(large_energies)

        fluence_large = f_flu(large_energies)

        # Still binning
        for ii, val in enumerate(large_energies):
            index = np.argmin(np.abs(original_energies_keV - val))
            weights_xray_small[index] += weights_xray[ii]
            weights_energies[index] += weights_e_long[ii]
            fluence_small[index] += fluence_large[ii]
        
        if hasattr(self, 'PCD'):
            if self.PCD == True:
                weights_energies /= original_energies_keV
                
        fluence_small /= np.sum(fluence_small)
        weights_xray_small /= np.sum(weights_xray_small)
        weights_energies /= np.sum(weights_energies)

        fluence_norm = spectra.y / np.sum(spectra.y)

        # ----------------------------------------------
        # -------- Scatter Correction ------------------
        # ----------------------------------------------
        if hasattr(self, 'scatter'):
            mc_scatter = np.load(os.path.join(data_path,"scatter",self.scatter))
            dist = self.scatter_coords
        else:
            if bowtie_on and kwargs["filter"][:3] == "bow":
                mc_scatter = np.load(
                    os.path.join(data_path, "scatter", "scatter_bowtie.npy")
                )
                dist = np.linspace(
                    -256 * 0.0784 - 0.0392, 256 * 0.0784 - 0.0392, 512
                )
                logging.info("   Scatter is filtered by bowtie")
            else:
                scatter = np.load(
                    os.path.join(data_path, "scatter", "scatter_updated.npy")
                )
                dist = np.linspace(
                    -256 * 0.0784 - 0.0392, 256 * 0.0784 - 0.0392, 512
                )

                def func(x, a, b):
                    return (-((152 / (np.sqrt(x ** 2 + 152 ** 2))) ** a)) * b

                mc_scatter = np.zeros(scatter.shape)

                for jj in range(mc_scatter.shape[1]):
                    popt, popc = curve_fit(
                        func, dist, scatter[:, jj], [10, scatter[256, jj]]
                    )
                    mc_scatter[:, jj] = func(dist, *popt)

        factor = (152 / (np.sqrt(dist ** 2 + 152 ** 2))) ** 3

        if ASG:
            # Modify the primary
            logging.info("Initializing ASG")
            flood_summed = (
                factor * 660 * 0.72
            )  # This is the 0.85 efficiency for the ASG

            lead = spectrum.get_mu(82)
            for jj in range(mc_scatter.shape[1]):
                # quarter will be let through well 0.75
                # will be filtered should be 39 microns
                # but made it bigger for the angle
                mc_scatter[:, jj] = 0.2 * mc_scatter[:, jj] + 0.8 * mc_scatter[
                    :, jj
                ] * np.exp(-0.007 * lead(original_energies_keV[jj]))
        else:
            flood_summed = factor * 660

        if bowtie_on:

            bowtie_coef = np.load(
                os.path.join(data_path, "filters", kwargs["filter"] + ".npy")
            )  # **1.1#**(2)*3
            flood_summed = (bowtie_coef.T) * flood_summed.squeeze()

        self.flood_summed = flood_summed

        def interpolate_pixel(series):

            # The scatter is 512 with 0.78 mm pixels
            logging.info(
                f"    Interp scatter 512 to {self.geomet.nDetector[1]} pixels"
            )
            no = 0.784
            npt = 512
            np2 = self.geomet.nDetector[1]
            no2 = self.geomet.dDetector[0]

            pts2 = np.linspace(0, np2 * no2, np2) - np2 / 2 * no2
            pts = np.linspace(0, npt * no, npt) - npt / 2 * no

            #             import ipdb;ipdb.set_trace()
            if len(series.shape) > 1:
                for ii in range(series.shape[0]):
                    series[ii] = np.interp(pts2, pts, series[ii])
            else:
                series = np.interp(pts2, pts, series)

            return series

        if (
            self.geomet.dDetector[0] != 0.784
            or self.geomet.dDetector[1] != 512
        ) and not hasattr(self, 'scatter'):
            flood_summed = interpolate_pixel(flood_summed)
            mc_scatter = interpolate_pixel(mc_scatter.T).T

        # ----------------------------------------------
        # -------- Ray Tracing -------------------------
        # ----------------------------------------------

        tile = True
        # The index of the different materials
        masks = np.zeros(
            [
                len(self.phan_map) - 1,
                self.phantom.shape[0],
                self.phantom.shape[1],
                self.phantom.shape[2],
            ]
        )
        mapping_functions = []

        # Get the mapping functions for the different
        #  tissues to reconstruct the phantom by energy
        for ii in range(1, len(self.phan_map)):
            mapping_functions.append(
                spectrum.get_mu(self.phan_map[ii].split(":")[0])
            )
            masks[ii - 1] = self.phantom == ii

        phantom2 = self.phantom.copy().astype(np.float32)
        doses = []

        intensity = np.zeros(
            [len(angles), self.geomet.nDetector[0], self.geomet.nDetector[1]]
        )
        noise = np.zeros(
            [len(angles), self.geomet.nDetector[0], self.geomet.nDetector[1]]
        )

        load_proj = False

        if load_proj:
            logging.info("Loading projections is on")
            projections = np.load(
                os.path.join(data_path, "raw_proj", "projections.npy")
            )

        logging.info("Running Simulations")
        
        if hasattr(kernel, 'fs_size'):
            logging.info(
                f"    {kernel.fs_size*kernel.pitch}"\
                " mm focal spot added"
            )

        for jj, energy in enumerate(original_energies_keV):
            
            # Change the phantom values
            if weights_xray_small[jj] == 0:
                doses.append(0)
                continue

            logging.info(f"    Simulating {energy} keV")

            for ii in range(0, len(self.phan_map) - 1):
                phantom2[masks[ii].astype(bool)] = mapping_functions[ii](
                    energy
                )

            if load_proj:
                projection = projections[jj]
            else:
                projection = self.ray_trace(phantom2, tile)

            kernel.kernels[jj + 1] /= np.sum(kernel.kernels[jj + 1])

            if bowtie_on:
                # Calculate a dose contribution by
                # dividing by 10 since tigre has projections
                # that are different
                # The bowtie is used to modify the dose absorbed by the filter
                doses.append(
                    np.mean(
                        (energy)
                        * (
                            -np.exp(-(projection * 0.997) / 10)
                            + bowtie_coef[:, jj]
                        )
                        * mu_en_water2[jj]
                        / mu_water2[jj]
                    )
                )
            else:
                #             logging.info('bowtie off')
                doses.append(
                    np.mean(
                        (energy)
                        * (1 - np.exp(-(projection * 0.997) / 10))
                        * mu_en_water2[jj]
                        / mu_water2[jj]
                    )
                )
            # Get the scale of the noise
            if bowtie_on:
                if scat_on:
                    int_temp = (
                        (
                            np.exp(-0.97 * np.array(projection) / 10)
                            * (flood_summed[jj])
                        )
                        + mc_scatter[:, jj]
                    ) * weights_xray_small[
                        jj
                    ]  # 0.97 JO
                else:
                    int_temp = (
                        (
                            np.exp(-0.97 * np.array(projection) / 10)
                            * (flood_summed[jj])
                        )
                    ) * weights_xray_small[
                        jj
                    ]  # 0.97 JO
            else:
                #                 int_temp =
                #  ((np.exp(-0.97*np.array(projection)/10)*(flood_summed)))
                # *weights_xray_small[jj]
                # # !!This took out the scatter for test
                if scat_on:
                    int_temp = (
                        (
                            np.exp(-0.97 * np.array(projection) / 10)
                            * (flood_summed)
                        )
                        + mc_scatter[:, jj]
                    ) * weights_xray_small[
                        jj
                    ]  # 0.97 JO
                else:
                    int_temp = (
                        (
                            np.exp(-0.97 * np.array(projection) / 10)
                            * (flood_summed)
                        )
                    ) * weights_xray_small[
                        jj
                    ]  # 0.97 J078

            noise_temp = np.random.poisson(np.abs(int_temp)) - int_temp

            (bx, by) = self.geomet.nDetector

            bx //= 1  # 16 # These are the boundaries for the convolution
            by //= 1  # 16 # This avoids some artifacts that creep in from
            # bounday effects

            if det_on and convolve_on:
                for ii in range(len(self.angles)):

                    if hasattr(kernel, 'fs_size'):

                        # foical spot with geometrical factor
                        fs_real = (
                            kernel.fs_size * self.geomet.DSO / self.geomet.DSD
                        )

                        mod_filter = gaussian_filter(
                            np.pad(
                                kernel.kernels[jj + 1], ((10, 10), (10, 10))
                            ),
                            fs_real,
                        )
                    else:
                        mod_filter = kernel.kernels[jj + 1]

                    int_temp[ii, bx:-bx, by:-by] = fftconvolve(
                        int_temp[ii, :, :], mod_filter, mode="same"
                    )[bx:-bx, by:-by]
                    noise_temp[ii, bx:-bx, by:-by] = fftconvolve(
                        noise_temp[ii, :, :],
                        kernel.kernels[jj + 1],
                        mode="same",
                    )[bx:-bx, by:-by]

            intensity += (
                int_temp * weights_energies[jj] / weights_xray_small[jj]
            )
            noise += noise_temp * weights_energies[jj] / weights_xray_small[jj]

        self.weights_small = weights_energies
        self.weights_small2 = weights_xray_small
        self.weights_small3 = weights_energies[jj] / weights_xray_small[jj]
        self.mc_scatter = mc_scatter

        logging.info("Weighting simulations")
        if not det_on:
            return intensity

        # ----------------------------------------------
        # ----------- Dose calculation -----------------
        # ----------------------------------------------

        # Sum over the image dimesions to get the energy
        # intensity and multiply by fluence, 2e7 was number
        # in reference MC simulations
        def get_dose_nphoton(nphot):
            return nphot / 2e7

        def get_dose_mgy(mgy, doses, fluence_small):
            nphoton = mgy / (
                get_dose_per_photon(doses, fluence_small)
                * (1.6021766e-13)
                * 1000
            )
            return get_dose_nphoton(nphoton)

        def get_dose_per_photon(doses, fluence_small):
            # linear fit of the data mod Mar 2021
            pp = np.array([0.87810143, 0.01136471])
            return ((np.array(doses) / 1000) @ (fluence_small)) * pp[0] + pp[1]

        ratio = None

        # Dose in micro grays
        if mgy != 0.0:
            ratio = get_dose_mgy(mgy, np.array(doses), fluence_small)
        elif nphoton is not None:
            ratio = get_dose_nphoton(nphoton)

        # --- Noise and Scatter Calculation ---
        # Now I interpolate deposition
        # and get the average photons reaching the detector
        deposition_long = np.interp(
            spectra.x,
            original_energies_keV,
            deposition[0] / (original_energies_keV / 1000) / 1000000,
        )
        nphotons_at_energy = fluence_norm * deposition_long

        self.nphotons_at = deposition_long
        nphotons_av = np.sum(nphotons_at_energy)

        if return_dose:
            pp = np.array([0.87810143, 0.01136471])
            return (
                np.array(doses),
                spectra.y,
                ((np.array(doses) / 1000) @ (fluence_small)),
                ((np.array(doses) / 1000) @ (fluence_small)) * pp[0] + pp[1],
            )

        # ----------------------------------------------
        # ----------- Add Noise ------------------------ # Delete? -Emily
        # ----------------------------------------------

        if ratio is not None:

            adjusted_ratio = ratio * nphotons_av
            logging.info(f"    Added noise {adjusted_ratio} times ref")
            # This is a moderately iffy approximation,
            # but probably not too bad except in the
            # large and small limit cases
            intensity = (
                intensity * adjusted_ratio
                + noise * (adjusted_ratio ** (1 / 2))
            ) / adjusted_ratio
            # Why did I do this again? 2021
        else:
            logging.info("    No noise was added")

        if return_intensity:
            return intensity

        if bowtie_on:
            flood_summed = (weights_energies) @ flood_summed

        self.proj = -10 * np.log(intensity / (flood_summed))

    def ray_trace(self, phantom2, tile):

        if self.tigre_works:  # resort to astra if tigre doesn't work
            try:
                return np.squeeze(tigre.Ax(phantom2, self.geomet, self.angles))
            except Exception:
                logging.info("WARNING: Tigre GPU not working")

                self.tigre_works = False

    def reconstruct(self, algo, filt="hamming"):

        if algo == "FDK":
            try:
                self.img = tigre.algorithms.fdk(
                    self.proj, self.geomet, self.angles, filter=filt
                )
            except Exception:
                logging.info("WARNING: Tigre failed during recon using Astra")
                self.img = self.astra_recon(self.proj.transpose([1, 0, 2]))

        if algo == "CGLS":
            try:
                self.img = tigre.algorithms.cgls(
                    self.proj.astype(np.float32),
                    self.geomet,
                    self.angles,
                    niter=20,
                )
            except Exception:
                logging.info("WARNING: Tigre failed during recon using Astra")
                self.img = self.astra_recon(self.proj.transpose([1, 0, 2]))

    def plot_projs(self, fig):

        subfig1 = fig.add_subplot(121)
        subfig2 = fig.add_subplot(122)

        tracker = IndexTracker(subfig1, self.proj.transpose([1, 2, 0]))
        fig.canvas.mpl_connect("scroll_event", tracker.onscroll)
        tracker2 = IndexTracker(subfig2, self.proj.transpose([0, 2, 1]))
        fig.canvas.mpl_connect("scroll_event", tracker2.onscroll)
        fig.tight_layout()

    def plot_recon(self, ind, vmin_max=None):
        """
        vmax_min; tuple, boundaries of cmap
        """
        plt.figure()

        if vmin_max is None:
            plt.imshow(self.img[ind], cmap="gray")
        else:
            plt.imshow(
                self.img[ind], cmap="gray", vmax=vmin_max[1], vmin=vmin_max[0]
            )

        plt.axis("equal")
        plt.axis("off")


class IndexTracker(object):
    """
    Shameless steal from matplotlib documentation
    Enables a plot that uses scrolling from cursor
    """

    def __init__(self, ax, X):
        self.ax = ax
        ax.set_title("Geometry Viewer")

        self.X = X
        rows, cols, self.slices = X.shape
        self.ind = self.slices // 2

        self.im = ax.imshow(self.X[:, :, self.ind])
        self.update()

    def onscroll(self, event):
        # logging.info("%s %s" % (event.button, event.step))
        if event.button == "up":
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        self.im.set_data(self.X[:, :, self.ind])
        self.ax.set_ylabel("slice %s" % self.ind)
        self.im.axes.figure.canvas.draw()
