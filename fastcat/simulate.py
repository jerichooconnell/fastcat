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
        return_intensity=False,
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
            return_intensity: Bool
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
        
#         return_intensity = False
        bowtie_on = False
        load_proj = False
        save_proj = False
        
        if "test" in kwargs.keys():
            if kwargs["test"] == 1:
                return_intensity = False
            if kwargs["test"] == 2:
                return_intensity = True

        if "verbose" in kwargs.keys():
            if kwargs["verbose"] == 0:
                level = logging.WARNING
                logger = logging.getLogger()
                logger.setLevel(level)
                for handler in logger.handlers:
                    handler.setLevel(level)
        
        if "bowtie" in kwargs.keys():
            if kwargs["bowtie"]:
                bowtie_on = True
                logging.info(f'Initializing filter {kwargs["filter"]}')
                
        if "load_proj" in kwargs.keys():
            if kwargs["load_proj"]:
                load_proj = True
                logging.info(f'Loading attenuations from {kwargs["proj_file"]}')
                
        if "save_proj" in kwargs.keys():
            if kwargs["save_proj"]:
                save_proj = True
                logging.info(f'Saving attenuations to {kwargs["proj_file"]}')
                
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
            logging.info("This is a small edep file starting at 30")
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
        
        spectra.x, spectra.y = spectra.get_points() #! This might throw errors!

        # Set the last value to zero so that
        # the linear interpolation doesn't mess up
        spectra.y[-1] = 0
        
        # This is the probability a photon incident on the detector will 
        # be detected times the energy of that photon.
        p_photon_detected_times_energy = (
            deposition[0] / (MC_energies_keV / 1000) / 1000000
        )
        
        # Binning to get the fluence per energy
        # Done on two keV intervals so that the rounding is even
        # between 10 keV bins 2 4 go down 6 8 go up. With 1 keV
        # it would lead to rounding errors
        long_energies_keV = np.linspace(0, 6000, 3001)
        f_fluence = interp1d(
            np.insert(spectra.x, (0, -1), (0, 6000)),
            np.insert(spectra.y, (0, -1), (0, spectra.y[-1])),
        )
        f_p_photon_detected_times_energy = interp1d(
            np.insert(MC_energies_keV, 0, 0),
            np.insert(p_photon_detected_times_energy, 0, 0),
        )
        f_energy = interp1d(
            np.insert(MC_energies_keV, 0, 0),
            np.insert((deposition[0]), 0, 0),
        )

        if not return_intensity:
            fluence_times_p_detected_times_energy_long = f_fluence(long_energies_keV) * f_p_photon_detected_times_energy(long_energies_keV)# * f_energy(long_energies_keV)
            fluence_times_energy_long = f_fluence(long_energies_keV) * f_energy(long_energies_keV)
        else:
            # If we return_intensity we only care about the counts reaching the detector
            # and not the probability of detection or the energy imparted through detection
            fluence_times_p_detected_times_energy_long = f_fluence(long_energies_keV)
            fluence_times_energy_long = f_fluence(long_energies_keV)
    
        fluence_long = f_fluence(long_energies_keV)

        # This is the integration step, a little clumsy but I wanted to get it right
        # so I didn't use trapz

        # These are the weights as they will be normalized
        # compared to the 
        w_fluence_times_p_detected_energy = np.zeros(len(MC_energies_keV))
        w_fluence_times_energy = np.zeros(len(MC_energies_keV))
        w_fluence = np.zeros(len(MC_energies_keV))
        
        for ii, val in enumerate(long_energies_keV):
            index = np.argmin(np.abs(MC_energies_keV - val))
            w_fluence_times_p_detected_energy[index] += fluence_times_p_detected_times_energy_long[ii]
            w_fluence_times_energy[index]     += fluence_times_energy_long[ii]
            w_fluence[index]                  += fluence_long[ii]
        
        # -----------------------------------------
        # -------------- PCD Detector -------------
        # -----------------------------------------
        # If the phantom has attribute PCD = True
        # the weighting is divided by the energy
        
        if hasattr(self, 'PCD'):
            if self.PCD == True:
                w_fluence_times_energy /= MC_energies_keV
        
        # Normalize all the weights
        w_fluence /= np.sum(w_fluence)
        w_fluence_times_p_detected_energy /= np.sum(w_fluence_times_p_detected_energy)
        w_fluence_times_energy /= np.sum(w_fluence_times_energy)

        # ----------------------------------------------
        # -------- Scatter Correction ------------------
        # ----------------------------------------------
        # Loads the scatter from file. The scatter can
        # be specified in the phantom. Otherwise it is the
        # default 10cm scatter or 10cm bowtie scatter from
        # file.
        # The scatter corresponds to 512 pixel detector
        # but will be rebinned later to detector specifications
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
        
        
        # Accounts for the curvature of the primary field on x dir
        curvature = (152 / (np.sqrt(dist ** 2 + 152 ** 2))) ** 3
        
        # -----------------------------------------------------
        # ----------------- SIMPLE ASG ------------------------
        #------------------------------------------------------
        # Modifies the relative contribution of the primary and
        # secondary by hardcoded curvatures
        
        if ASG:
            # Modify the primary
            logging.info("Initializing ASG")
            flood_field_profile = (
                curvature * 660 * 0.72
            )  # This is the 0.85 efficiency for the ASG

            lead = spectrum.get_mu(82)
            for jj in range(-len(MC_energies_keV),0):
                # quarter will be let through well 0.75
                # will be filtered should be 39 microns
                # but made it bigger for the angle
                mc_scatter[:, jj] = 0.2 * mc_scatter[:, jj] + 0.8 * mc_scatter[
                    :, jj
                ] * np.exp(-0.007 * lead(MC_energies_keV[jj]))
        else:
            flood_field_profile = curvature * 660
        
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
        if bowtie_on:
            bowtie_coef = np.load(
                os.path.join(data_path, "filters", kwargs["filter"] + ".npy")
            ) 
            flood_field_profile = ((bowtie_coef.T) * flood_field_profile.squeeze())[-len(MC_energies_keV):]
            
#         self.intensities = []
        self.flood_field_profile = flood_field_profile
        
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
            flood_field_profile = interpolate_pixel(flood_field_profile)
            mc_scatter = interpolate_pixel(mc_scatter.T).T

        tile = True # Legacy variable that doesn't do anything
        
        # ----------------------------------------------------
        # ----------- Phantom Initialization -----------------
        # ----------------------------------------------------
        # We have to configure the materials in the phantom so
        # that the materials attenuation updates according to the energy.
        
        # The masks are quite memory intensive and provide the index of
        # each material in the phantom but I don't want to find
        # the materials each time so I make the 4D array
        masks = np.zeros(
            [
                len(self.phan_map) - 1,
                self.phantom.shape[0],
                self.phantom.shape[1],
                self.phantom.shape[2],
            ]
        )
        energy2mu_functions = []
        doses = []
        
        # Get the functions that return the linear attenuation coefficient
        # of each of the materials in the phantom and put them in this list
        # attenuation coefficients come from the csv files in data/mu/
        for ii in range(1, len(self.phan_map)):
            energy2mu_functions.append(
                spectrum.get_mu(self.phan_map[ii].split(":")[0])
            )
            masks[ii - 1] = self.phantom == ii

        phantom2 = self.phantom.copy().astype(np.float32)

        intensity = np.zeros(
            [len(angles), self.geomet.nDetector[0], self.geomet.nDetector[1]]
        )
        noise = np.zeros(
            [len(angles), self.geomet.nDetector[0], self.geomet.nDetector[1]]
        )

        if load_proj:
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
        
        for jj, energy in enumerate(MC_energies_keV):
            
            # We don't simulate energies that have no
            # fluence so this ignores these energies
            if w_fluence_times_p_detected_energy[jj] == 0:
                doses.append(0)
                continue

            logging.info(f"    Simulating {energy} keV")
            
            # Update the phantom attenuation values based
            # on the energy.
            for ii in range(0, len(self.phan_map) - 1):
                phantom2[masks[ii].astype(bool)] = energy2mu_functions[ii](
                    energy
                )
            
            # ----------- Load attenuation or Save ----------------------------
            # Here we give the option to load the attenuations from files in the
            # raw proj directory, this can be good if you keep on raytracing the
            # smae phantom without modifying the geometry or materials of the simulation
            # but changing the spectrum and the detector
            if load_proj:
                attenuation = np.load(
                    os.path.join(user_data_path, "raw_proj", kwargs['proj_file']+'_'+energy+'.npy')
                )
            else:
                attenuation = self.ray_trace(phantom2, tile)
                
            if save_proj:
                np.save(
                    os.path.join(user_data_path, "raw_proj", kwargs['proj_file']+'_'+energy+'.npy',attenuation)
                )

            kernel.kernels[jj + 1] /= np.sum(kernel.kernels[jj + 1])
            
            # ------------------------------------------------------------
            # ------------- Dose Calculation -----------------------------
            # ------------------------------------------------------------
            # Here we calculate the energy deposited by the specific beam
            # in the phantom approximating the phantom as water and using
            # the mass energy absorbtion coefficient for the energy
            # We then use an empirical relation to the MC dose to get a
            # dose estimate. It is a bit confusing and is in the fastcat paper
            
            if bowtie_on:
                # Calculate a dose contribution by
                # dividing by 10 since tigre has attenuations
                # that are different
                # The bowtie is used to modify the dose absorbed by the filter
                doses.append(
                    np.mean(
                        (energy)
                        * (
                            -np.exp(-(attenuation * 0.997) / 10)
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
            if bowtie_on:
                if scat_on:
                    int_temp = (
                        (
                            np.exp(-0.97 * np.array(attenuation) / 10)
                            * (flood_field_profile[jj])
                        )
                        + mc_scatter[:, jj]
                    ) * w_fluence_times_p_detected_energy[
                        jj
                    ]
                else:
                    int_temp = (
                        (
                            np.exp(-0.97 * np.array(attenuation) / 10)
                            * (flood_field_profile[jj])
                        )
                    ) * w_fluence_times_p_detected_energy[
                        jj
                    ]  # 0.97 JO
            else:
                if scat_on:
                    int_temp = (
                        (
                            np.exp(-0.97 * np.array(attenuation) / 10)
                            * (flood_field_profile)
                        )
                        + mc_scatter[:, jj]
                    ) * w_fluence_times_p_detected_energy[
                        jj
                    ]  # 0.97 JO
                else:
                    int_temp = (
                        (
                            np.exp(-0.97 * np.array(attenuation) / 10)
                            * (flood_field_profile)
                        )
                    ) * w_fluence_times_p_detected_energy[
                        jj
                    ]  # 0.97 J078
            
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
            
            noise_temp = np.random.poisson(np.abs(int_temp)) - int_temp

            (bx, by) = self.geomet.nDetector

            bx //= 1  # 16 # These are the boundaries for the convolution
            by //= 1  # 16 # This avoids some artifacts that creep in from
            # bounday effects

            if not return_intensity and convolve_on:
                for ii in range(len(self.angles)):

                    if hasattr(kernel, 'fs_size'):

                        # foical spot with geometrical curvature
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
                    
            # ------------------------------------------------------
            # ---------- Now we remove the detector weighting ------
            # ------------------------------------------------------
            # I'm not sure why I do this but usually these changes came
            # from fitting analytical results.
            
            if return_intensity:
                intensity += (
                    int_temp * w_fluence_times_energy[jj] / w_fluence_times_p_detected_energy[jj]
                )
                noise += noise_temp * w_fluence_times_energy[jj] / w_fluence_times_p_detected_energy[jj]
            else:
                intensity += int_temp #* energy/np.sum(MC_energies_keV)
                noise += noise_temp #* energy/np.sum(MC_energies_keV)
            
            # I want the edep in MeV
#             self.intensities.append(int_temp * w_fluence_times_energy[jj] / w_fluence_times_p_detected_energy[jj])

        self.weights_small = w_fluence_times_energy
        self.weights_small2 = (w_fluence_times_energy/ w_fluence_times_p_detected_energy)/np.nansum(w_fluence_times_energy / w_fluence_times_p_detected_energy)
        self.weights_small3 = w_fluence_times_p_detected_energy
        self.mc_scatter = mc_scatter
        self.dep = deposition[0]

        logging.info("Weighting simulations")
        
        # This shouldn't work as the profiles should be weighted by the detector response already?
        if return_intensity:
            return intensity

        # ----------------------------------------------
        # ----------- Dose calculation -----------------
        # ----------------------------------------------

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

        # Dose in micro grays
        if mgy != 0.0:
            ratio = get_dose_mgy(mgy, np.array(doses), w_fluence)
        elif nphoton is not None:
            ratio = get_dose_nphoton(nphoton)

        # --- Noise and Scatter Calculation ---
        # Now I interpolate deposition
        # and get the average photons reaching the detector
        p_detected_times_energy_long = np.interp(
            spectra.x,
            MC_energies_keV,
            deposition[0] / (MC_energies_keV / 1000) / 1000000,
        )
                
        # Use the long normalized fluence
        fluence_norm_long = spectra.y / np.sum(spectra.y)
        nphotons_at_energy = fluence_norm_long * p_detected_times_energy_long

        nphotons_av = np.sum(nphotons_at_energy)
        self.nphoton_av = nphotons_av
        self.nphotons_at = np.array(doses)@w_fluence
        self.ratio = ratio
        
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

        if ratio is not None:

            adjusted_ratio = ratio * nphotons_av
            logging.info(f"    Added noise {adjusted_ratio} times reference")
            intensity = (
                intensity * adjusted_ratio
                + noise * (adjusted_ratio ** (1 / 2))
            ) / adjusted_ratio
        else:
            logging.info("    No noise was added")
        
        if bowtie_on:
            flood_field_profile = (w_fluence_times_energy) @ flood_field_profile

        self.proj = -10 * np.log(intensity / (flood_field_profile))
        
        # Check for bad values
        self.proj[~np.isfinite(self.proj)] = 1000
        
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
