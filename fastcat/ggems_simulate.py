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
import pickle
import sys

import numpy as np
import tigre
from scipy.ndimage import gaussian_filter, laplace
from scipy.optimize import curve_fit
from scipy.signal import fftconvolve
from scipy.interpolate import interp1d
from scipy.stats import poisson
from scipy.ndimage import zoom

from fastcat.ggems_scatter import save_ggems_simulation_parameters

try:
    import matplotlib.pyplot as plt
    PLOT_AVAILABLE = True
except ImportError:
    PLOT_AVAILABLE = False

from fastcat import spectrum, utils

logging.basicConfig(
    stream=sys.stdout,
    format="[%(asctime)s] {%(filename)s:%"
    "(lineno)d} %(levelname)s - %(message)s",
    level=logging.DEBUG,
)
# Set the logger to be more verbose for only this module

# logging.getLogger('fastcat.simulate').setLevel(logging.DEBUG)

__author__ = "Jericho OConnell"
__version__ = "0.0.1"

data_path = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "data")
user_data_path = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "user_data")

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

    def simulate(
        self,
        kernel,
        spectra,
        angles,
        nphoton=None,
        mgy=0.0,
        return_dose=False,
        test_intensity=False,
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
            test_intensity: Bool
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
                transmission curvature of 0.72 and a scatter
                transmission curvature of 0.32.
                Default False (off)
            filter: string
                String which specifies one of the
                filters in data/filters/ these filter
                the beam before it hits the phantom.
            silent: Bool
                True silences the logger.

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

        # ------------------------------------------
        # ---------- Dealing with kwargs -----------
        # ------------------------------------------
        # kwargs map to a few variables that I set
        self.simulated = False
        self.filter_on = False
        self.load_proj = False
        self.save_proj = False
        self.fast_noise = False
        self.mgy = mgy
        self.ASG = ASG
        self.convolve_on = convolve_on

        if "test" in kwargs.keys():
            if kwargs["test"] == 1:
                test_intensity = False
            if kwargs["test"] == 2:
                test_intensity = True

        if "silent" in kwargs.keys():
            if kwargs["silent"]:
                level = logging.WARNING
                logger = logging.getLogger()
                logger.setLevel(level)
                for handler in logger.handlers:
                    handler.setLevel(level)

        if "bowtie" in kwargs.keys():
            self.filter_on = kwargs["bowtie"]
            logging.info(
                f'Initializing filter {self.bowtie_file}')

        if "self.load_proj" in kwargs.keys():
            self.load_proj = kwargs["self.load_proj"]
            logging.info(
                f'Loading attenuations from {kwargs["proj_file"]}')

        if "self.save_proj" in kwargs.keys():
            self.save_proj = kwargs["self.save_proj"]
            logging.info(
                f'Saving attenuations to {kwargs["proj_file"]}')

        if "self.fast_noise" in kwargs.keys():
            self.fast_noise = kwargs["self.fast_noise"]

        if test_intensity:
            # This will return photon counts rather than energy detected
            # photon counts is what 'intensity' refers to
            logging.info('    Detector is photon counting')
            self.PCD = True

        self.tigre_works = tigre_works
        self.angles = angles
        deposition = np.load(
            kernel.deposition_efficiency_file, allow_pickle=True
        )

        # These are the energies that simulations take place at
        # they match what I used in the Monte Carlo
        # legacy energies don't include below 30 keV
        if len(deposition[0]) == 18:
            MC_energies_keV = np.array(
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
            logging.info(
                "This is a small edep file starting at 30")
            MC_energies_keV = np.array(
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

        # ----------------------------------------------------------
        # ------------ Making the weights --------------------------
        # ----------------------------------------------------------
        # Since the energies are not uniformly distributed
        # We have to do a coarse integration to get the relative
        # fluence at different energies
        if hasattr(spectra, 'get_points'):
            # ! This might throw errors!
            spectra.x, spectra.y = spectra.get_points()
        else:
            # ! This might throw errors!Aluminum
            spectra.x, spectra.y = spectra.get_spectrum()

        # Set the last value to zero so that
        # the linear interpolation doesn't mess up
        spectra.y[-1] = 0

        # This is the probability a photon incident on the detector will
        # be detected times the energy of that photon.
        p_photon_detected_times_energy = (
            deposition[0] /
            (MC_energies_keV / 1000) / 1000000
        )
        # Binning to get the fluence per energy
        # Done on two keV intervals so that the rounding is even
        # between 10 keV bins 2 4 go down 6 8 go up. With 1 keV
        # it would lead to rounding errors
        long_energies_keV = np.linspace(0, 6000, 3001)
        f_fluence = interp1d(
            np.insert(spectra.x, (0, -1), (0, 6000)),
            np.insert(spectra.y, (0, -1),
                      (0, spectra.y[-1])),
        )

        fluence_long = f_fluence(long_energies_keV)

        # This is the integration step, a little clumsy but I wanted to get it right
        # so I didn't use trapz

        # These are the weights as they will be normalized
        # compared to the
        w_fluence = np.zeros(len(MC_energies_keV))

        for ii, val in enumerate(long_energies_keV):
            index = np.argmin(np.abs(MC_energies_keV - val))
            w_fluence[index] += fluence_long[ii]

        # Fluence is normalized, it only weights the composition of
        # a reference beam that has about 660 photons per pixels
        # using this reference we can know the noise and dose
        w_fluence /= np.sum(w_fluence)

        self.weights_fluence = w_fluence

#         return

        if not test_intensity:
            w_fluence_times_p_detected_energy = w_fluence * \
                deposition[0]
            w_fluence_times_p_detected = w_fluence_times_p_detected_energy / \
                MC_energies_keV
        else:
            # If we test_intensity we only care about the counts reaching the detector
            # and not the probability of detection or the energy imparted through detection
            w_fluence_times_p_detected_energy = w_fluence
            w_fluence_times_p_detected = w_fluence

        # Normalize all the weights
        weight_scale = np.sum(w_fluence_times_p_detected)
        w_fluence_times_p_detected /= weight_scale
        w_fluence_times_p_detected_energy /= np.sum(
            w_fluence_times_p_detected_energy)

        self.weight_scale = weight_scale
        # ----------------------------------------------
        # -------- Adding Scatter ------------------
        # ----------------------------------------------
        # Loads the scatter from file. The scatter can
        # be specified in the phantom. Otherwise it is the
        # default 10cm scatter or 10cm bowtie scatter from
        # file.
        # The scatter corresponds to 512 pixel detector
        # but will be rebinned later to detector specifitions
        if hasattr(self, 'scatter'):
            mc_scatter = np.load(os.path.join(
                data_path, "scatter", self.scatter))
            dist = self.scatter_coords
            logging.info(
                f'    Loading scatter from file {self.scatter}')
        else:
            # if self.filter_on and kwargs["filter"][:3] == "bow":
            #     mc_scatter = np.load(
            #         os.path.join(
            #             data_path, "scatter", "scatter_bowtie.npy")
            #     )
            #     dist = np.linspace(
            #         -256 * 0.0784 - 0.0392, 256 * 0.0784 - 0.0392, 512
            #     )
            #     logging.info(
            #         "   Scatter is modified by bowtie")
            # else:

            # Pixel positions based off the geometry
            dist = np.linspace(- self.geomet.nDetector[0] / 2 * self.geomet.dDetector[0] - self.geomet.dDetector[0] / 2,
                               self.geomet.nDetector[0] / 2 *
                               self.geomet.dDetector[0] -
                               self.geomet.dDetector[0] / 2,
                               self.geomet.nDetector[0])

        # Accounts for the curvature of the primary field on x dir
        curvature = np.ones_like(dist)

        # -----------------------------------------------------
        # ----------------- SIMPLE ASG ------------------------
        # ------------------------------------------------------
        # Modifies the relative contribution of the primary and
        # secondary by hardcoded curvatures

        if ASG:
            # Modify the primary
            logging.info("Initializing ASG")
            flood_photon_counts = (
                curvature * 660 * 0.72
            )  # This is the 0.85 efficiency for the ASG

            lead = spectrum.get_mu(82)
            for jj in range(-len(MC_energies_keV), 0):
                # quarter will be let through well 0.75
                # will be filtered should be 39 microns
                # but made it bigger for the angle
                mc_scatter[:, jj] = 0.2 * mc_scatter[:, jj] + 0.8 * mc_scatter[
                    :, jj
                ] * np.exp(-0.007 * lead(MC_energies_keV[jj]))
        else:
            flood_photon_counts = curvature * 660

        # ---------------------------------------------------
        # -------------- Bowtie Initialization --------------
        # ---------------------------------------------------
        # Modifies the primary by the attenuation of the bowtie
        # which is loaded from a data file. Does this for each
        # energy

        # If the bowtie is on, rather than the flood profile being
        # one dimensional it will get an added dimension of energy.
        # This messes with some stuff, so later you might see a
        # different handling of bowtie scatter and projections.
        def radial_distance(shape):
            pixel_size = self.geomet.dDetector[0]
            x, y = np.meshgrid(
                np.arange(shape[0]), np.arange(shape[1]))
            x = x - shape[0] / 2 - \
                self.geomet.offDetector[0]
            y = y - shape[1] / 2 - \
                self.geomet.offDetector[1]
            x = x * pixel_size
            y = y * pixel_size
            return np.sqrt(x**2 + y**2)

        # Make an array measuring the intensity in the projection as a function of radial distance
        def r_squared_falloff():
            shape = self.geomet.nDetector
            r = radial_distance(shape)
            return (self.geomet.DSD / (r**2 + self.geomet.DSD**2)**0.5)**3

        geometric_falloff = r_squared_falloff()

        flood_photon_counts = np.ones_like(
            geometric_falloff[0])
        # If the bowtie is on, rather than the flood profile being
        # one dimensional it will get an added dimension of energy.
        # This messes with some stuff, so later you might see a
        # different handling of bowtie scatter and projections.
        if self.filter_on:
            bowtie_coef = self.generate_flood_field(
                self.bowtie_file, interp=self.bowtie_interp)
            flood_photon_counts = (
                (bowtie_coef.T) * flood_photon_counts.squeeze())[-len(MC_energies_keV):]
            self.bowtie_coef = bowtie_coef
            # JO Not sure if this is right
            # self.bowtie_coef *= MC_energies_keV
            # print('mutliplying by energy')

        self.bowtie_coef = bowtie_coef
        self.flood_photon_counts = flood_photon_counts

        # --------------------------------------------
        # --------- Adjust flood and Scatter Size ----
        # --------------------------------------------
        # If the detector is not 512 pixels in the axial
        # direction we interpolate using the pixel dimension
        # and number of pixels. Modifies flood field and
        # mc scatter

        # def interpolate_pixel(series):

        #     # The scatter is 512 with 0.78 mm pixels
        #     logging.info(
        #         f"    Interp scatter 512 to {self.geomet.nDetector[1]} pixels"
        #     )
        #     no = 0.784
        #     npt = 512
        #     np2 = self.geomet.nDetector[1]
        #     no2 = self.geomet.dDetector[0]

        #     pts2 = np.linspace(
        #         0, np2 * no2, np2) - np2 / 2 * no2
        #     pts = np.linspace(
        #         0, npt * no, npt) - npt / 2 * no

        #     # Make a blank array to interpolate into

        #     if len(series.shape) > 1:
        #         series2 = np.zeros(
        #             [series.shape[0], np2]
        #         )
        #         for ii in range(series.shape[0]):
        #             series2[ii] = np.interp(
        #                 pts2, pts, series[ii])

        #         series = series2
        #     else:
        #         series = np.interp(pts2, pts, series)

        #     return series

        # if (
        #     self.geomet.dDetector[0] != 0.784
        #     or self.geomet.dDetector[1] != 512
        # ) and not hasattr(self, 'scatter'):
        #     flood_photon_counts = interpolate_pixel(
        #         flood_photon_counts)
        #     mc_scatter = interpolate_pixel(mc_scatter.T).T

        if not self.filter_on:
            # flood profile is the same for all energies without filter
            flood_photon_counts = np.tile(
                flood_photon_counts, [len(MC_energies_keV), 1])

        self.flood_photon_counts = flood_photon_counts
        self.energies = MC_energies_keV
        self.w_fluence_times_p_detected = w_fluence_times_p_detected

        tile = True  # Legacy variable that doesn't do anything
        self.intensities = []
        # self.mc_scatter = mc_scatter
        # ----------------------------------------------------
        # ----------- Phantom Initialization -----------------
        # ----------------------------------------------------
        # We have to configure the materials in the phantom so
        # that the materials attenuation updates according to the energy.

        # The masks are quite memory intensive and provide the index of
        # each material in the phantom but I don't want to find
        # the materials each time so I make the 4D array
        if self.is_non_integer:
            uniques = np.unique(self.phantom)
            masks = np.zeros(
                [
                    len(uniques),
                    self.phantom.shape[0],
                    self.phantom.shape[1],
                    self.phantom.shape[2],
                ], dtype=bool
            )
        else:
            masks = np.zeros(
                [
                    len(self.phan_map) - 1,
                    self.phantom.shape[0],
                    self.phantom.shape[1],
                    self.phantom.shape[2],
                ], dtype=bool
            )
        energy2mu_functions = []
        doses = []

        # Get the functions that return the linear attenuation coefficient
        # of each of the materials in the phantom and put them in this list
        # attenuation coefficients come from the csv files in data/mu/
        if self.is_non_integer:
            uniques = np.unique(self.phantom)
            for jj, ii in enumerate(uniques):
                masks[jj - 1] = self.phantom == ii
        else:
            for ii in range(1, len(self.phan_map)):
                masks[ii - 1] = self.phantom == ii

        if hasattr(self, 'from_nrrd'):
            for ii in range(1, len(self.phan_map)):
                energy2mu_functions.append(
                    spectrum.get_mu_over_rho(
                        self.phan_map[ii].split(":")[0])
                )
        else:
            for ii in range(1, len(self.phan_map)):
                energy2mu_functions.append(
                    spectrum.get_mu(
                        self.phan_map[ii].split(":")[0])
                )

        phantom2 = self.phantom.copy().astype(np.float32)

        intensity = np.zeros(
            [len(angles), self.geomet.nDetector[0],
             self.geomet.nDetector[1]]
        )
        noise = np.zeros(
            [len(angles), self.geomet.nDetector[0],
             self.geomet.nDetector[1]]
        )
        flood_energy_abs = np.zeros_like(
            flood_photon_counts[0])

        if self.load_proj:
            logging.info("Loading attenuations is on")

        logging.info("Running Simulations")

        if hasattr(kernel, 'fs_size'):
            logging.info(
                f"    {kernel.fs_size*kernel.pitch}"
                " mm focal spot added"
            )

        # ----------------------------------------------
        # -------- Ray Tracing -------------------------
        # ----------------------------------------------
        # This is the main step for ray tracing where
        # TIGRE is called and the projections are created.
        # TIGRE gives the attenuation along a ray from
        # source to detector.

        if self.fast_noise:
            logging.info(
                '    Fast Noise algo! Beware of innacurate results')

        for jj, energy in enumerate(MC_energies_keV):

            # We don't simulate energies that have no
            # fluence so this ignores these energies
            if w_fluence[jj] == 0:
                doses.append(0)
                continue

            logging.info(f"    Simulating {energy} keV")

            # Update the phantom attenuation values based
            # on the energy.
            if self.is_non_integer:
                '''
                This is a function to split the voxel as in between two tissue
                if it is in the middle
                '''
                for ii, unique in enumerate(uniques[1:]):
                    if unique.is_integer():
                        # print('unique',unique)
                        phantom2[masks[ii]] = energy2mu_functions[int(unique)-1](
                            energy
                        )
                    else:
                        # print('non unique',unique,len(energy2mu_functions))
                        phantom2[masks[ii]] = ((1-unique % 1) *
                                               energy2mu_functions[int(np.floor(unique-1))](energy) +
                                               (unique % 1)*energy2mu_functions[int(
                                                   np.ceil(unique-1))](energy)
                                               )
            else:
                for ii in range(0, len(self.phan_map) - 1):
                    phantom2[masks[ii]] = energy2mu_functions[ii](
                        energy
                    )

            # ----------- Load attenuation or Save ----------------------------
            # Here we give the option to load the attenuations from files in the
            # raw proj directory, this can be good if you keep on raytracing the
            # smae phantom without modifying the geometry or materials of the simulation
            # but changing the spectrum and the detector
            if self.load_proj:
                attenuation = np.load(
                    os.path.join(user_data_path, "raw_proj",
                                 kwargs['proj_file']+'_'+f'{energy}'+'.npy')
                )
            else:
                if hasattr(self, 'from_nrrd'):
                    attenuation = self.ray_trace(
                        (phantom2*self.density).astype(np.float32), tile)
                else:
                    attenuation = self.ray_trace(
                        phantom2, tile)

            if self.save_proj:
                np.save(
                    os.path.join(user_data_path, "raw_proj", kwargs['proj_file']+'_'+f'{energy}'+'.npy'), attenuation)

            kernel.kernels[jj +
                           1] /= np.sum(kernel.kernels[jj + 1])

            # ------------------------------------------------------------
            # ------------- Dose Calculation -----------------------------
            # ------------------------------------------------------------
            # Here we calculate the energy deposited by the specific beam
            # in the phantom approximating the phantom as water and using
            # the mass energy absorbtion coefficient for the energy
            # We then use an empirical relation to the MC dose to get a
            # dose estimate. It is a bit confusing and is in the fastcat paper

            # if self.filter_on:
            #     # Calculate a dose contribution by
            #     # dividing by 10 since tigre has attenuations
            #     # that are different
            #     # The bowtie is used to modify the dose absorbed by the filter
            #     doses.append(
            #         np.mean(
            #             (energy)
            #             * bowtie_coef[:, jj]
            #             * (1
            #                 - np.exp(-(attenuation * 0.997) / 10)
            #                )
            #             * mu_en_water2[jj]
            #             / mu_water2[jj]
            #         )
            #     )
            # else:
            #     #             logging.info('bowtie off')
            #     doses.append(
            #         np.mean(
            #             (energy)
            #             * (1 - np.exp(-(attenuation * 0.997) / 10))
            #             * mu_en_water2[jj]
            #             / mu_water2[jj]
            #         )
            #     )

            # --------------------------------------------------
            # ---------- Attenuation to Intensity ---------------
            # --------------------------------------------------
            # We use the flood field profile to get intensity from
            # the attenuation. We need intensity to calculate the
            # noise and convolve with PSF.

            # Intensity temp at this point is weighted by fluence
            # energy and the detector response which is what the
            # detector sees.
#             if self.filter_on:
            if scat_on:
                int_temp = (
                    (
                        np.exp(-0.97 * attenuation / 10)
                        * (flood_photon_counts[jj])
                    )
                    + mc_scatter[:, jj]
                ) * w_fluence_times_p_detected[
                    jj
                ]
            else:
                int_temp = (
                    (
                        np.exp(-0.97 * attenuation / 10)
                        * (flood_photon_counts[jj])
                    )
                ) * w_fluence_times_p_detected[
                    jj
                ]
            # Mutliply by geometric factors
            int_temp *= geometric_falloff

            # 0.97 JO
#             else:
#                 if scat_on:
#                     int_temp = (
#                         (
#                             np.exp(-0.97 * attenuation / 10)
#                             * (flood_photon_counts[jj])
#                         )
#                         + mc_scatter[:, jj]
#                     ) * w_fluence_times_p_detected[
#                         jj
#                     ]  # 0.97 JO
#                 else:
#                     int_temp = (
#                         (
#                             np.exp(-0.97 * attenuation / 10)
#                             * (flood_photon_counts[jj])
#                         )
#                     ) * w_fluence_times_p_detected[
#                         jj
#                     ]  # 0.97 J078

            # --------------------------------------------------
            # ---------- Get the noise -------------------------
            # --------------------------------------------------
            # The noise is based on the dose. The dose can't be calculated
            # until all attenuations are calculated when we find
            # the relative dose contributions for each energy.
            # We could save each set of attenuations and combine them
            # after the dose calculation but that is brutal on memory
            # To avoid this we make a reference noise that weights the
            # noise according to energy and fluence and scale it later.

            # Generate random numbers along axial slice and then choose from
            # a sample of 200 rather than generating a million poisson
            # variables, may not want to do this in some cases.
#             self.fast_noise = True #False #True

            if self.fast_noise:
                noise_first = np.random.poisson(np.abs(np.mean(
                    np.mean(int_temp[:, :, :], 0), 0)), [400, int_temp.shape[2]])
                noise_temp = noise_first[np.random.choice(noise_first.shape[0],
                                                          int_temp.shape[:2])] - int_temp
            else:
                noise_temp = poisson.rvs(
                    np.abs(int_temp)) - int_temp

            # ------------------------------------------
            # --- Convert Counts to Energy Absorbed ----
            # ------------------------------------------
            # The photon counts are multiplied by their energy to get
            # the contribution to the detector.

            if hasattr(self, 'PCD'):
                if self.PCD:
                    flood_energy_abs_temp = flood_photon_counts[jj] * \
                        w_fluence_times_p_detected[jj]

            else:
                noise_temp *= energy
                int_temp *= energy
                flood_energy_abs_temp = flood_photon_counts[jj] * \
                    w_fluence_times_p_detected[jj]*energy

            (bx, by) = self.geomet.nDetector
            bx //= 1  # 16 # These are the boundaries for the convolution
            by //= 1  # 16 # This avoids some artifacts that creep in from
            # bounday effects

            if not test_intensity and convolve_on:
                for ii in range(len(self.angles)):

                    if hasattr(kernel, 'fs_size'):
                        # focal spot with geometrical magnification
                        fs_real = (
                            kernel.fs_size * self.geomet.DSO / self.geomet.DSD
                        )
                        mod_filter = gaussian_filter(
                            np.pad(
                                kernel.kernels[jj +
                                               1], ((10, 10), (10, 10))
                            ),
                            fs_real,
                        )
                    else:
                        mod_filter = kernel.kernels[jj + 1]

                    int_temp[ii] = fftconvolve(
                        int_temp[ii, :,
                                 :], mod_filter, mode="same"
                    )
                    noise_temp[ii] = fftconvolve(
                        noise_temp[ii, :, :],
                        kernel.kernels[jj + 1],
                        mode="same",
                    )

            # ------------------------------------------------------
            # ---------- Now we remove the detector weighting ------
            # ------------------------------------------------------
            # I'm not sure why I do this but usually these changes came
            # from fitting analytical results.

            # * energy/np.sum(MC_energies_keV)
            intensity += int_temp
            # * energy/np.sum(MC_energies_keV)
            noise += noise_temp
            flood_energy_abs += flood_energy_abs_temp

            # I want the edep in MeV
            # if test_intensity:
            #     self.intensities.append(int_temp)
#         self.mod_filter = mod_filter
        logging.info("Weighting simulations")

        self.flood_field = geometric_falloff * flood_energy_abs

        # Normalize the flood field to the mean
        mean_ff = np.mean(self.flood_field)
        self.flood_field /= mean_ff

        # Calculate how many photons per pixel
        # The number of photons is the number of photons reaching the detector
        # To get the photons per pixel we divide by the number of pixels
        npixels = self.geomet.nDetector[0] * \
            self.geomet.nDetector[1]

        photons_per_pixel = nphoton / npixels

        # Normalize the flood_field to the number of photons per pixel
        self.flood_field *= photons_per_pixel

        # Normalize the intensity to the flood field
        intensity /= mean_ff
        intensity *= photons_per_pixel

        # Normalize the noise to the flood field
        noise /= mean_ff
        noise *= photons_per_pixel**(1/2) # Poisson noise-ish

        # ----------------------------------------------
        # ----------- Dose calculation -----------------
        # ----------------------------------------------
        # This shouldn't work as the profiles should be weighted by the detector response already?
        # if test_intensity:
        #     return intensity + noise

        self.intensity = intensity + noise
        # Sum over the image dimesions to get the energy
        # intensity and multiply by fluence, 2e7 was number
        # in reference MC simulations
        # def get_dose_nphoton(nphot):
        #     return nphot / 2e7

        # def get_dose_mgy(mgy, doses, w_fluence):
        #     nphoton = mgy / (
        #         get_dose_per_photon(doses, w_fluence)
        #         * (1.6021766e-13)
        #         * 1000
        #     )
        #     return get_dose_nphoton(nphoton)

        # def get_dose_per_photon(doses, w_fluence):
        #     # linear fit of the data mod Mar 2021
        #     pp = np.array([0.87810143, 0.01136471])
        #     return ((np.array(doses) / 1000) @ (w_fluence)) * pp[0] + pp[1]

        # ratio = None

        # self.doses = doses
        # # Dose in micro grays
        # if mgy != 0.0:
        #     ratio = get_dose_mgy(
        #         mgy, np.array(doses), w_fluence)
        # elif nphoton is not None:
        #     ratio = get_dose_nphoton(nphoton)

        # --- Noise and Scatter Calculation ---
        # Now I interpolate deposition
        # and get the average photons reaching the detector
        # p_detected_times_energy_long = np.interp(
        #     spectra.x,
        #     MC_energies_keV,
        #     deposition[0] /
        #     (MC_energies_keV / 1000) / 1000000,
        # )

        # # Use the long normalized fluence
        # fluence_norm_long = spectra.y / np.sum(spectra.y)
        # nphotons_at_energy = fluence_norm_long * \
        #     p_detected_times_energy_long

        # nphotons_av = np.sum(nphotons_at_energy)
        # self.nphoton_av = nphotons_av
        # self.nphotons_at = np.array(doses)@w_fluence
        # self.ratio = ratio

        # if return_dose:
        #     pp = np.array([0.87810143, 0.01136471])
        #     return (
        #         np.array(doses),
        #         spectra.y,
        #         ((np.array(doses) / 1000) @ (w_fluence)),
        #         ((np.array(doses) / 1000) @
        #          (w_fluence)) * pp[0] + pp[1],
        #     )

        # # ----------------------------------------------
        # # ----------- Add Noise ------------------------
        # # ----------------------------------------------

        # if ratio is not None:

        #     # The noise scales in quadrature while the intensity
        #     # scales linearly to give the right noise level

        #     # The real noise level is the ratio of the calculated dose
        #     # to the reference dose data from MC times nphotons_av which
        #     # is a factor accounting for the detector efficiency.
        #     adjusted_ratio = ratio * nphotons_av
        #     logging.info(
        #         f"    Added noise {adjusted_ratio} times reference")
        #     intensity = (
        #         intensity * adjusted_ratio
        #         + noise * (adjusted_ratio) ** (1 / 2)
        #     ) / adjusted_ratio
        # else:
        #     logging.info("    No noise was added")

        # return_intensity = True

        # if return_intensity:
        #     self.intensity = intensity
        #     self.flood = flood_energy_abs

        # self.proj = -10 * \
        #     np.log(intensity / (flood_energy_abs))

        # # Check for bad values
        # self.proj[~np.isfinite(self.proj)] = 1000

    def get_scatter_contrib(self):
        '''
        Small function for returning the scatter intensity
        profile from fastcat simulation

        Can only be run after a simulation has been performed
        otherwise the variables won't be recognized
        '''

        scatter_int_profile = np.zeros(
            self.flood_photon_counts[0].shape[0])

        for jj in range(len(self.w_fluence_times_p_detected)):

            scatter_int_profile += (
                self.mc_scatter[:, jj] *
                self.w_fluence_times_p_detected[jj] *
                self.energies[jj]
            )

        return scatter_int_profile

    def add_poisson_noise(self, amplification=1):
        '''
        This is a function that adds noise to the intensity
        profile
        '''
        # Add poisson noise to the intensity profile
        self.intensity = poisson.rvs(
            amplification*np.abs(self.intensity))
        self.ggems_scatter_denoised = poisson.rvs(
            amplification*np.abs(self.ggems_scatter_denoised))

    def detector_mtf_convolution(self, kernel):

        if hasattr(kernel, 'fs_size'):
            # focal spot with geometrical magnification
            fs_real = (
                kernel.fs_size * self.geomet.DSO / self.geomet.DSD
            )
            mod_filter = gaussian_filter(
                kernel.kernel,
                fs_real,
            )
        else:
            mod_filter = kernel.kernel

        mod_filter /= np.sum(mod_filter)

        for ii in range(len(self.angles)):
            self.intensity[ii, :, :] = fftconvolve(
                self.intensity[ii, :,
                               :], mod_filter, mode="same"
            )
            self.interpolated_ggems_scatter[ii, :, :] = fftconvolve(
                self.interpolated_ggems_scatter[ii, :, :],
                mod_filter,
                mode="same",
            )

    def invert_fastcat_parameters(self, kernel):

        self.improved_images = np.zeros_like(self.intensity)

        for ii in range(self.intensity.shape[0]):

            denoised = denoise_tv_chambolle(projection)
            deconvolved_RL = richardson_lucy(
                denoised, kernel.kernel, iterations=30, clip=False)
            scatter = phantom.get_scatter_contrib()
            self.improved[ii] = deconvolved_RL - scatter

    def ray_trace(self, phantom2, tile):

        if self.tigre_works:  # resort to astra if tigre doesn't work
            return np.squeeze(tigre.Ax(phantom2, self.geomet, self.angles))

    def load_ggems_files(self, ggems_scatter_files, ggems_primary_files, flood_file,
                         average_scatter=False):

        # Define the denoising functions
        def downsize_block(image, block_size):
            image_size = image.shape[0]
            block_size = int(block_size)
            downsize = image_size//block_size
            image = image[:downsize *
                          block_size, :downsize*block_size]
            image = image.reshape(downsize, block_size, downsize,
                                  block_size).mean(axis=(1, 3))
            return image

        def denoise_projections(projections):
            projections_zoomed = []

            projections = np.array(projections)
            # Check if the array is 3D
            if projections.ndim == 2:
                projections = projections[np.newaxis, :, :]
            for projection in projections:
                projection_zoomed_downsized = downsize_block(
                    projection, 64)
                projection_zoomed = zoom(
                    projection_zoomed_downsized, 64, order=4, mode='nearest')
                # Adjust the mean of the zoomed image to match the original
                projection_zoomed = projection_zoomed * \
                    np.mean(projection) / \
                    np.mean(projection_zoomed)
                projections_zoomed.append(projection_zoomed)

            return np.array(projections_zoomed)

        ggems_primary_projections = []
        ggems_scatter_projections = []

        for ggems_scatter_file, ggems_primary_file in zip(ggems_scatter_files, ggems_primary_files):
            ggems_primary, b, c = utils.read_mhd(
                ggems_primary_file)
            ggems_scatter, b, c = utils.read_mhd(
                ggems_scatter_file)

            # Weird transformation lol
            ggems_scatter_projections.append(
                np.rot90(np.fliplr(ggems_scatter.squeeze()), 1))
            ggems_primary_projections.append(np.rot90(np.fliplr(
                ggems_primary.squeeze()), 1) - ggems_scatter_projections[-1])
        # Get the ggems projections

        def average_ggems_scatter(projections):
            # If radially symmetric, average the scatter
            sum_scat = np.mean(
                projections, axis=0)
            for ii in range(projections.shape[0]):
                projections[ii] = sum_scat

            return projections

        self.ggems_primary_projections = np.array(
            ggems_primary_projections)
        self.ggems_scatter_projections = np.array(
            ggems_scatter_projections)
        if average_scatter:
            self.ggems_scatter_projections = average_ggems_scatter(
                self.ggems_scatter_projections)
        self.ggems_scatter_denoised = denoise_projections(
            self.ggems_scatter_projections)

        # Load the flood field
        ggems_flood, b, c = utils.read_mhd(flood_file)
        ggems_flood = ggems_flood.squeeze()

        self.ggems_flood = ggems_flood

    def calc_ggems_projections(self, scat_on=True):

        ggems_projections = []

        for i in range(self.intensity.shape[0]):
            # Calculate the projections using beer lambert law and the ggems scatter
            if scat_on:
                ggems_projections.append(- np.log((
                    self.intensity[i] + self.interpolated_ggems_scatter[i])
                    / self.flood_field))
            else:
                ggems_projections.append(- np.log((
                    self.intensity[i])
                    / self.flood_field))

        self.ggems_projections = np.array(ggems_projections)

    def correct_intensity(self, crop=20, ml=True):
        from sklearn.neural_network import MLPRegressor
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler

        # Normalize with a crop
        def normalize_mean_and_std(array1, array2, crop):
            # Check if the array is 3D
            if array1.ndim == 2:
                array1 = array1 - \
                    np.mean(array1[crop:-crop, crop:-crop])
                array1 = array1 * \
                    np.std(array2[crop:-crop, crop:-crop]) / \
                    np.std(array1[crop:-crop, crop:-crop])
                array1 = array1 + \
                    np.mean(array2[crop:-crop, crop:-crop])
                return array1
            else:
                array1 = array1 - \
                    np.mean(
                        array1[:, crop:-crop, crop:-crop])
                array1 = array1 * np.std(array2[:, crop:-crop, crop:-crop]) / np.std(
                    array1[:, crop:-crop, crop:-crop])
                array1 = array1 + \
                    np.mean(
                        array2[:, crop:-crop, crop:-crop])
                return array1

        def normalize_mean(arrays, array2, crop):
            factor = np.mean(arrays[0][:, crop:-crop, crop:-crop]) / \
                np.mean(array2[:, crop:-crop, crop:-crop])
            arrays[0] = arrays[0] / factor
            arrays[1] = arrays[1] / factor
            return arrays[0], arrays[1]

        # Normalize the intensity and the flood field flood field is 10e10 and these are 10e11
        self.intensity, self.flood_field = normalize_mean(
            [self.intensity, self.flood_field], self.ggems_primary_projections, crop=crop)
        # self.flood_field = normalize_mean_and_std(
        #     self.flood_field, self.ggems_flood * 1e11/1e10, crop=crop)

        # Load the ggems projection
        if ml:
            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(
                np.rot90(self.intensity[0, crop:-crop, crop:-crop], 3).flatten()[::4], self.ggems_primary_projections[0, crop:-crop, crop:-crop].flatten()[::4], test_size=0.2, random_state=42)

            # Scale the data
            scaler = StandardScaler()
            scaler.fit(X_train.reshape(-1, 1))
            X_train_scaled = scaler.transform(
                X_train.reshape(-1, 1))

            # # Define the neural network
            mlp = MLPRegressor(hidden_layer_sizes=(10, 10, 10),
                               activation='relu',
                               solver='adam',
                               max_iter=100,
                               verbose=False,
                               random_state=42)

            # Train the neural network
            mlp.fit(X_train_scaled, y_train)

            for ii, projection in enumerate(self.intensity):
                projection_scaled = scaler.transform(
                    projection.flatten().reshape(-1, 1))
                projection_nn = mlp.predict(
                    projection_scaled)
                projection_nn = projection_nn.reshape(
                    projection.shape)
                self.intensity[ii] = projection_nn

            # Scale the flood
            ff_scaled = scaler.transform(
                self.flood_field.flatten().reshape(-1, 1))
            ff_nn = mlp.predict(
                ff_scaled)
            ff_nn = ff_nn.reshape(
                self.flood_field.shape)
            self.flood_field = ff_nn

    def interpolate_ggems_scatter(self):
        image = self.ggems_scatter_denoised
        z1 = self.angles
        z0 = np.deg2rad(self.sim_angles)
        # Check if its in degs or radians
        if np.max(z0) > 2 * np.pi:
            z0 = z0 / 180 * np.pi
        # loop over the pixels and interpolate the images
        self.interpolated_ggems_scatter = np.zeros(
            (len(z1), image.shape[1], image.shape[2]))
        for ii in range(image.shape[1]):
            for jj in range(image.shape[2]):
                self.interpolated_ggems_scatter[:, ii, jj] = np.interp(
                    z1, z0, image[:, ii, jj])

    def reconstruct(self, algo, filt="hamming"):

        if algo == "FDK":
            try:
                self.img = tigre.algorithms.fdk(
                    self.proj, self.geomet, self.angles, filter=filt
                )
            except Exception:
                logging.info(
                    "WARNING: Tigre failed during recon using Astra")
                self.img = self.astra_recon(
                    self.proj.transpose([1, 0, 2]))

        if algo == "CGLS":
            try:
                self.img = tigre.algorithms.cgls(
                    self.proj.astype(np.float32),
                    self.geomet,
                    self.angles,
                    niter=20,
                )
            except Exception:
                logging.info(
                    "WARNING: Tigre failed during recon using Astra")
                self.img = self.astra_recon(
                    self.proj.transpose([1, 0, 2]))

    def reconstruct_ggems(self, algo, filt="hamming"):

        if algo == "FDK":
            try:
                self.img = tigre.algorithms.fdk(
                    self.ggems_projections, self.geomet, self.angles, filter=filt
                )
            except Exception:
                logging.info(
                    "WARNING: Tigre failed during recon using Astra")
                self.img = self.astra_recon(
                    self.proj.transpose([1, 0, 2]))

        if algo == "CGLS":
            try:
                self.img = tigre.algorithms.cgls(
                    self.ggems_projections.astype(
                        np.float32),
                    self.geomet,
                    self.angles,
                    niter=20,
                )
            except Exception:
                logging.info(
                    "WARNING: Tigre failed during recon using Astra")
                self.img = self.astra_recon(
                    self.proj.transpose([1, 0, 2]))
    # def from_file(self, file_name):
    #     # Load a phantom from a pickle file
    #     with open(file_name, "rb") as f:
    #         self.img = pickle.load(f)

    def plot_projs(self, fig):

        subfig1 = fig.add_subplot(121)
        subfig2 = fig.add_subplot(122)

        tracker = IndexTracker(
            subfig1, self.proj.transpose([1, 2, 0]))
        fig.canvas.mpl_connect(
            "scroll_event", tracker.onscroll)
        tracker2 = IndexTracker(
            subfig2, self.proj.transpose([0, 2, 1]))
        fig.canvas.mpl_connect(
            "scroll_event", tracker2.onscroll)
        fig.tight_layout()

    def from_pickle(self, file_name):
        # Load a phantom from a pickle file
        with open(file_name, "rb") as f:
            return pickle.load(f)

    def generate_ggems_bash_script(self, output_dir, detector_material,
                                   nparticles, spectrum=None, s_max=None,
                                   edep_detector=False, dli=False, **kwargs):
        '''
        Generate a bash script to give the scatter at angles
        '''

        fname = save_ggems_simulation_parameters(self, output_dir, detector_material,
                                                 nparticles, spectrum, s_max, edep_detector, dli, **kwargs)

        # Write a bash script with each entry calling ggems_scatter script
        # with a different angle

        with open(os.path.join(output_dir, 'ggems_scatter.sh'), 'w') as f:
            # Write the header
            f.write('#!/bin/bash\n\n')
            # Write a for loop over the angles
            for ii, angle in enumerate(self.angles):
                f.write(
                    f'python ggems_scatter_script.py {fname} --angle {angle} --number {ii}\n')

        self.bash_scipt = os.path.join(
            output_dir, 'ggems_scatter.sh')

    def save_ggems_simulation_parameters(self, output_dir, detector_material,
                                         nparticles, spectrum=None, s_max=None,
                                         edep_detector=False, dli=False, **kwargs):
        '''
        Saves the parameters of the ggems simulation into a dictionary in the simulation
        object. This is used to save the parameters of the simulation to a pickle file
        '''
        simulation_parameters = {}
        # simulation_parameters['output_file'] = output_file
        # simulation_parameters['output_dir'] = output_dir
        simulation_parameters['detector_material'] = detector_material
        simulation_parameters['nparticles'] = nparticles
        simulation_parameters['spectrum'] = spectrum
        simulation_parameters['s_max'] = s_max
        simulation_parameters['edep_detector'] = edep_detector
        simulation_parameters['dli'] = dli
        # simulation_parameters['phantom'] = phantom
        simulation_parameters['kwargs'] = kwargs

        output_file = os.path.join(
            output_dir, f'ggems_{f"{nparticles:.0e}".replace("+", "")}_{s_max:.0f}kVp')

        self.simulation_parameters = simulation_parameters

        # Create a new directory for the simulation
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Print the data to be saved
        print(f"Data to be saved: {simulation_parameters}")
        with open(os.path.join(output_file + '.pkl'), 'wb') as f:
            pickle.dump(self, f)
            print('Done saving simulation parameters to ' +
                  os.path.join(output_file + '.pkl'))

        return output_file + '.pkl'

    def save_pickle(self, output_file):
        # Save the phantom to a pickle file
        with open(output_file, "wb") as f:
            pickle.dump(self, f)
    # Create a function that return the flood field

    def generate_flood_field(self, bowtie_file, interp='nearest'):
        # Based on the geometry find the angle of each pixel from the source
        # Define the source position (z)
        source_position = self.geomet.DSD
        # Define the pixel position in the detector plane
        pixel_position = np.linspace(-self.geomet.sDetector[0]/2,
                                     self.geomet.sDetector[0]/2, self.geomet.nDetector[0])
        # Calculate the angle of each pixel from the source
        pixel_angles = np.rad2deg(
            np.arctan(pixel_position/source_position))
        # For each pixel angle find the bowtie filter thickness
        # Load the bowtie filter
        # Define a function to parse the bowtie filter

        def parse_bowtie_filter(bowtie_file):
            # read the bowtie file
            with open(bowtie_file, 'r') as f:
                bowtie_filter = f.read()
            bowtie_filter_columns = {}
            # Split the bowtie filter into lines
            bowtie_filter_lines = bowtie_filter.split('\n')
            bowtie_filter_columns['material'] = bowtie_filter_lines[2]
            bowtie_filter_columns['density'] = float(
                bowtie_filter_lines[4])
            bowtie_filter_columns['angular_increment'] = float(
                bowtie_filter_lines[6])
            bowtie_filter_columns['start_angle'] = float(
                bowtie_filter_lines[8])
            bowtie_filter_columns['thickness'] = np.array(
                bowtie_filter_lines[10:-1]).astype(float)
            bowtie_filter_columns['angles'] = np.arange(bowtie_filter_columns['start_angle'], bowtie_filter_columns['start_angle'] +
                                                        bowtie_filter_columns['angular_increment'] * len(bowtie_filter_columns['thickness']), bowtie_filter_columns['angular_increment'])
            return bowtie_filter_columns

        bowtie_dict = parse_bowtie_filter(bowtie_file)

        def calculate_bowtie_thicknesses(bowtie_dict, pixel_angles, interp='nearest'):
            # Interpolate the bowtie filter thicknesses to the pixel angles using nearest neighbour interpolation
            if interp == 'nearest':
                bowtie_thicknesses = np.interp(
                    pixel_angles, bowtie_dict['angles'], bowtie_dict['thickness'], left=0, right=0)
            if interp == 'linear':
                bowtie_thicknesses = np.interp(
                    pixel_angles, bowtie_dict['angles'], bowtie_dict['thickness'])
            if interp == 'cubic':
                bowtie_thicknesses = interp1d(
                    bowtie_dict['angles'], bowtie_dict['thickness'], kind='cubic', fill_value=0, bounds_error=False)(pixel_angles)

            return bowtie_thicknesses

        bowtie_by_angle = calculate_bowtie_thicknesses(
            bowtie_dict, pixel_angles, interp=interp)

        # Make an array that is the radial distance from the center of the image
        # def radial_distance(image, phantom):
        #     pixel_size = phantom.geomet.dDetector[0]
        #     x, y = np.meshgrid(
        #         np.arange(image.shape[0]), np.arange(image.shape[1]))
        #     x = x - image.shape[0] / 2 - \
        #         phantom.geomet.offDetector[0]
        #     y = y - image.shape[1] / 2 - \
        #         phantom.geomet.offDetector[1]
        #     x = x * pixel_size
        #     y = y * pixel_size
        #     return np.sqrt(x**2 + y**2)

        # Loop through the energies and calculate the attenuation
        MC_energies_keV = np.array(
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

        al_interp = spectrum.get_mu('13')
        filter_coefs = []

        for energy in MC_energies_keV:
            al_coef = al_interp(energy)
            filter_coefs.append(
                np.exp(-(bowtie_by_angle/13)*al_coef))  # Should be 10 to convert to mm, but 13 matched MC better

        return np.array(filter_coefs).T

    def run_ggems_bash_script(self):
        # Check if the bash script exists
        if not hasattr(self, 'bash_scipt'):
            logging.info('No bash script to run')
            return

        # Run the bash script
        os.system(self.bash_scipt)

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
