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
from scipy.ndimage import gaussian_filter, laplace
from scipy.optimize import curve_fit
from scipy.signal import fftconvolve
from scipy.interpolate import interp1d
from scipy.stats import poisson

from fastcat.ggems_scatter import run_ggems_scatter_simulation


try:
    import matplotlib.pyplot as plt
    PLOT_AVAILABLE = True
except ImportError:
    PLOT_AVAILABLE = False

from fastcat.spectrum import Spectrum

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

data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
user_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "user_data")

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
        
        # Sort out the kwargs
        self.sort_kwargs(kwargs)
        self.mgy = mgy
        self.ASG = ASG
        self.convolve_on = convolve_on
        self.test_intensity = test_intensity
        self.tigre_works = tigre_works
        self.angles = angles
        deposition = np.load(
            kernel.deposition_efficiency_file, allow_pickle=True
        )

        # These are the energies that simulations take place at
        # they match what I used in the Monte Carlo simulations
        self.MC_energies_keV = np.array(
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
        
        # ----------------------------------------------------------
        # ------------ Making the weights --------------------------
        # ----------------------------------------------------------
        # Since the energies are not uniformly distributed
        # We have to do a coarse integration to get the relative
        # fluence at different energies
        if hasattr(spectra,'get_points'):
            spectra.x, spectra.y = spectra.get_points() #! This might throw errors!
        else:
            spectra.x, spectra.y = spectra.get_spectrum() #! This might throw errors!
            
        # Set the last value to zero so that
        # the linear interpolation doesn't mess up
        spectra.y[-1] = 0
        
        # Binning to get the fluence per energy
        # Done on two keV intervals so that the rounding is even
        # between 10 keV bins 2 4 go down 6 8 go up. With 1 keV
        # it would lead to rounding errors
        long_energies_keV = np.linspace(0, 6000, 3001)

        f_fluence = interp1d(
            np.insert(spectra.x, (0, -1), (0, 6000)),
            np.insert(spectra.y, (0, -1), (0, spectra.y[-1])),
        )
    
        fluence_long = f_fluence(long_energies_keV)

        # This is the integration step, a little clumsy but I wanted to get it right
        # so I didn't use trapz

        # These are the weights as they will be normalized
        # compared to the 
        w_fluence = np.zeros(len(self.MC_energies_keV))
        
        for ii, val in enumerate(long_energies_keV):
            index = np.argmin(np.abs(self.MC_energies_keV - val))
            w_fluence[index] += fluence_long[ii]
        
        # Fluence is normalized, it only weights the composition of 
        # a reference beam that has about 660 photons per pixels
        # using this reference we can know the noise and dose
        w_fluence /= np.sum(w_fluence)
        
        self.weights_fluence = w_fluence
    
        if not self.test_intensity:
            w_fluence_times_p_detected_energy = w_fluence*deposition[0]
            w_fluence_times_p_detected = w_fluence_times_p_detected_energy / self.MC_energies_keV
        else:
            # If we self.test_intensity we only care about the counts reaching the detector
            # and not the probability of detection or the energy imparted through detection
            w_fluence_times_p_detected_energy = w_fluence
            w_fluence_times_p_detected = w_fluence

        
        # Normalize all the weights
        weight_scale = np.sum(w_fluence_times_p_detected)
        w_fluence_times_p_detected /= weight_scale
        w_fluence_times_p_detected_energy /= np.sum(w_fluence_times_p_detected_energy)
        
        self.weight_scale = weight_scale
        
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
        if self.filter_on:
            bowtie_coef = np.load(
                os.path.join(data_path, "filters", kwargs["filter"] + ".npy")
            ) 
            flood_photon_counts = ((bowtie_coef.T) * flood_photon_counts.squeeze())[-len(self.MC_energies_keV):]
            self.bowtie_coef = bowtie_coef
        
        self.flood_photon_counts = flood_photon_counts
        
        # --------------------------------------------
        # --------- Adjust flood and Scatter Size ----
        # --------------------------------------------
        # If the detector is not 512 pixels in the axial
        # direction we interpolate using the pixel dimension
        # and number of pixels. Modifies flood field and 
        # mc scatter
        
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

            # Make a blank array to interpolate into

            if len(series.shape) > 1:
                series2 = np.zeros(
                    [series.shape[0], np2]
                )
                for ii in range(series.shape[0]):
                    series2[ii] = np.interp(pts2, pts, series[ii])
                
                series = series2
            else:
                series = np.interp(pts2, pts, series)

            return series

        if (
            self.geomet.dDetector[0] != 0.784
            or self.geomet.dDetector[1] != 512
        ) and not hasattr(self, 'scatter'):
            flood_photon_counts = interpolate_pixel(flood_photon_counts)
            mc_scatter = interpolate_pixel(mc_scatter.T).T

        if not self.filter_on:
            #flood profile is the same for all energies without filter
            flood_photon_counts = np.tile(flood_photon_counts,[len(self.MC_energies_keV),1])
            
        self.flood_photon_counts = flood_photon_counts
        self.energies = self.MC_energies_keV
        self.w_fluence_times_p_detected = w_fluence_times_p_detected
            
        tile = True # Legacy variable that doesn't do anything
        self.intensities = [] 
        self.mc_scatter = mc_scatter
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
            for jj,ii in enumerate(uniques):
                masks[jj - 1] = self.phantom == ii
        else:
            for ii in range(1, len(self.phan_map)):
                masks[ii - 1] = self.phantom == ii
        for ii in range(1, len(self.phan_map)):
            energy2mu_functions.append(
                spectrum.get_mu(self.phan_map[ii].split(":")[0])
            )

        phantom2 = self.phantom.copy().astype(np.float32)

        intensity = np.zeros(
            [len(angles), self.geomet.nDetector[0], self.geomet.nDetector[1]]
        )
        noise = np.zeros(
            [len(angles), self.geomet.nDetector[0], self.geomet.nDetector[1]]
        )
        flood_energy_abs = np.zeros_like(flood_photon_counts[0])

        if self.load_proj:
            logging.info("Loading attenuations is on")

        logging.info("Running Simulations")
        
        if hasattr(kernel, 'fs_size'):
            logging.info(
                f"    {kernel.fs_size*kernel.pitch}"\
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
            logging.info('    Fast Noise algo! Beware of innacurate results') 
                
        for jj, energy in enumerate(self.MC_energies_keV):
            
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
                for ii,unique in enumerate(uniques[1:]):
                    if unique.is_integer(): 
                        phantom2[masks[ii]] = energy2mu_functions[int(unique)-1](
                            energy
                        )
                    else:
                        phantom2[masks[ii]] = ((1-unique%1) * 
                        energy2mu_functions[int(np.floor(unique-1))](energy) + 
                        (unique%1)*energy2mu_functions[int(np.ceil(unique-1))](energy)
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
                    os.path.join(user_data_path, "raw_proj", kwargs['proj_file']+'_'+f'{energy}'+'.npy')
                )
            else:
                attenuation = self.ray_trace(phantom2, tile)
                
            if self.save_proj:
                np.save(
                    os.path.join(user_data_path, "raw_proj", kwargs['proj_file']+'_'+f'{energy}'+'.npy'),attenuation)

            kernel.kernels[jj + 1] /= np.sum(kernel.kernels[jj + 1])
            
            # ------------------------------------------------------------
            # ------------- Dose Calculation -----------------------------
            # ------------------------------------------------------------
            # Here we calculate the energy deposited by the specific beam
            # in the phantom approximating the phantom as water and using
            # the mass energy absorbtion coefficient for the energy
            # We then use an empirical relation to the MC dose to get a
            # dose estimate. It is a bit confusing and is in the fastcat paper
             
            if self.filter_on:
                # Calculate a dose contribution by
                # dividing by 10 since tigre has attenuations
                # that are different
                # The bowtie is used to modify the dose absorbed by the filter
                doses.append(
                    np.mean(
                        (energy)
                        * bowtie_coef[:, jj]
                        * (1 
                            -np.exp(-(attenuation * 0.997) / 10)
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
                        * (1 - np.exp(-(attenuation * 0.997) / 10))
                        * mu_en_water2[jj]
                        / mu_water2[jj]
                    )
                )
            
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
            
            # ------------------------------------------
            # --- Convert Counts to Energy Absorbed ----
            # ------------------------------------------
            # The photon counts are multiplied by their energy to get
            # the contribution to the detector.
            
            if hasattr(self, 'PCD'):
                if self.PCD:
                    flood_energy_abs_temp = flood_photon_counts[jj]*w_fluence_times_p_detected[jj]

            else:
                noise_temp *= energy
                int_temp *= energy
                flood_energy_abs_temp = flood_photon_counts[jj]*w_fluence_times_p_detected[jj]*energy
            
            (bx, by) = self.geomet.nDetector
            bx //= 1  # 16 # These are the boundaries for the convolution
            by //= 1  # 16 # This avoids some artifacts that creep in from
            # bounday effects

            if not self.test_intensity and convolve_on:
                for ii in range(len(self.angles)):

                    if hasattr(kernel, 'fs_size'):
                        # focal spot with geometrical magnification
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

                    int_temp[ii] = fftconvolve(
                        int_temp[ii, :, :], mod_filter, mode="same"
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
            
            intensity += int_temp
            noise += noise_temp
            flood_energy_abs += flood_energy_abs_temp
        
            # I want the edep in MeV
            if self.test_intensity:
                self.intensities.append(int_temp)
#         self.mod_filter = mod_filter
        logging.info("Weighting simulations")

        # ----------------------------------------------
        # ----------- Dose calculation -----------------
        # ----------------------------------------------
        # This shouldn't work as the profiles should be weighted by the detector response already?
        if self.test_intensity:
            return intensity
        
        self.get_dose(mgy, doses, w_fluence, nphoton)
        
        if return_dose:
            pp = np.array([0.87810143, 0.01136471])
            return (
                np.array(doses),
                spectra.y,
                ((np.array(doses) / 1000) @ (w_fluence)),
                ((np.array(doses) / 1000) @ (w_fluence)) * pp[0] + pp[1],
            )

        # ----------------------------------------------
        # ----------- Add Noise ------------------------
        # ----------------------------------------------

        if hasattr(self, 'ratio'):
            # The noise scales in quadrature while the intensity
            # scales linearly to give the right noise level
            
            # The real noise level is the ratio of the calculated dose
            # to the reference dose data from MC times nphotons_av which
            # is a factor accounting for the detector efficiency.
            adjusted_ratio = self.ratio * self.nphotons_av
            logging.info(f"    Added noise {adjusted_ratio} times reference")
            intensity = (
                intensity * adjusted_ratio
                + noise * (adjusted_ratio) ** (1 / 2)
            ) / adjusted_ratio
        else:
            logging.info("    No noise was added")
        
        return_intensity = True
        
        if return_intensity:
            self.intensity = intensity
            self.flood = flood_energy_abs
        
        self.proj = -10 * np.log(intensity / (flood_energy_abs))
        
        # Check for bad values
        self.proj[~np.isfinite(self.proj)] = 1000

    def get_dose(self, mgy, doses, w_fluence,nphoton):
        # Sum over the image dimesions to get the energy
        # intensity and multiply by fluence, 2e7 was number
        # in reference MC simulations
        def get_dose_nphoton(nphot):
            return nphot / 2e7
        
        def get_dose_mgy(mgy, doses, w_fluence):
            nphoton = mgy / (
                get_dose_per_photon(doses, w_fluence)
                * (1.6021766e-13)
                * 1000
            )
            return get_dose_nphoton(nphoton)

        def get_dose_per_photon(doses, w_fluence):
            # linear fit of the data mod Mar 2021
            pp = np.array([0.87810143, 0.01136471])
            return ((np.array(doses) / 1000) @ (w_fluence)) * pp[0] + pp[1]

        ratio = None
        
        self.doses = doses
        # Dose in micro grays
        if mgy != 0.0:
            ratio = get_dose_mgy(mgy, np.array(doses), w_fluence)
        elif nphoton is not None:
            ratio = get_dose_nphoton(nphoton)

        # --- Noise and Scatter Calculation ---
        # Now I interpolate deposition
        # and get the average photons reaching the detector
        p_detected_times_energy_long = np.interp(
            self.spectra.x,
            self.MC_energies_keV,
            self.deposition[0] / (self.MC_energies_keV / 1000) / 1000000,
        )
                
        # Use the long normalized fluence
        fluence_norm_long = self.spectra.y / np.sum(self.spectra.y)
        nphotons_at_energy = fluence_norm_long * p_detected_times_energy_long

        nphotons_av = np.sum(nphotons_at_energy)
        self.nphoton_av = nphotons_av
        self.nphotons_at = np.array(doses)@w_fluence
        self.ratio = ratio

    def get_ggems_scatter(self, nparticles=1e5, edep=False, s_max=100, angle = 0):

        run_ggems_scatter_simulation(self,
                                    nparticles=int(nparticles),
                                    output_file='',
                                    output_dir=self.out_dir,
                                    spectrum=self.spectrum,
                                    detector_material='CsI',
                                    vis=False,
                                    edep_detector=edep,
                                    s_max=s_max, angle = angle)
        
    def sort_kwargs(self, kwargs):
        # ------------------------------------------
        # ---------- Dealing with kwargs -----------
        # ------------------------------------------
        # kwargs map to a few variables that I set
        self.simulated = False
        self.filter_on = False
        self.load_proj = False
        self.save_proj = False
        self.fast_noise = False
        
        if "test" in kwargs.keys():
            if kwargs["test"] == 1:
                self.test_intensity = False
            if kwargs["test"] == 2:
                self.test_intensity = True

        if "silent" in kwargs.keys():
            if kwargs["silent"]:
                level = logging.WARNING
                logger = logging.getLogger()
                logger.setLevel(level)
                for handler in logger.handlers:
                    handler.setLevel(level)
        
        if "bowtie" in kwargs.keys():
            self.filter_on = kwargs["bowtie"]
            logging.info(f'Initializing filter {kwargs["filter"]}')
                
        if "self.load_proj" in kwargs.keys():
            self.load_proj = kwargs["self.load_proj"]
            logging.info(f'Loading attenuations from {kwargs["proj_file"]}')
                
        if "self.save_proj" in kwargs.keys():
            self.save_proj = kwargs["self.save_proj"]
            logging.info(f'Saving attenuations to {kwargs["proj_file"]}')
                
        if "self.fast_noise" in kwargs.keys():
            self.fast_noise = kwargs["self.fast_noise"]
                
        if self.test_intensity:
            # This will return photon counts rather than energy detected
            # photon counts is what 'intensity' refers to
            logging.info('    Detector is photon counting')
            self.PCD = True        
        
    def get_scatter_contrib(self):
        '''
        Small function for returning the scatter intensity
        profile from fastcat simulation
        
        Can only be run after a simulation has been performed
        otherwise the variables won't be recognized
        '''
        
        scatter_int_profile = np.zeros(self.flood_photon_counts[0].shape[0])
        
        for jj in range(len(self.w_fluence_times_p_detected)):
            
            scatter_int_profile += (
                self.mc_scatter[:,jj]*
                self.w_fluence_times_p_detected[jj] *
                self.energies[jj]
            )
        
        return scatter_int_profile
        
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

    