#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
Fastcat a rapid and accurate CBCT simulator.

PHANTOMS: CODE CONTAINING PHANTOM PARAMETERS AND ANALYSIS

"""

from __future__ import print_function

import glob
import logging
import os

import numpy as np
from numpy import cos, sin

from fastcat.ggems_simulate import Phantom
from fastcat.utils import get_phantom_from_mhd, nrrd_to_mhd, get_phan_map_from_range
from fastcat.fastmc_scatter import write_fastmc_xml_file, run_fastmc_files, write_fastmc_flood_field_xml_file
from fastcat.detector import Detector
from fastcat.spectrum import Spectrum

data_path = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "data")


class patient_phantom(Phantom):
    def __init__(self, nrrd_file, nparticles=None, is_fullfan=True, geo=None, reload=False, **kwargs):
        '''
        Create a patient phantom from an nrrd file and loads
        it as a phantom object

        nrrd_file: str
            The nrrd file to be used as the patient phantom
        nparticles: int
            The number of particles to be simulated
        is_fullfan: bool
            If the bowtie file is for full fan mode
        geo: tigre.geometry
            The geometry of the phantom
        reload: bool
            If the phantom should be reloaded from a pickle file in the directory

        kwargs: dict
            sim_num: int
                The simulation number to be reloaded
            info: bool
                If the list of available simulations should be printed
            second_layer: bool
                If the phantom has a second layer use the ggems files from the second layer

        '''
        nrrd_dir = nrrd_file.split('/')[-1].split(".")[0]

        if reload:
            logging.info("Reloading phantom")
            if os.path.exists(os.path.join(data_path, "user_phantoms", nrrd_dir)):
                logging.info(
                    f"Found existing directory. Using mhd file {nrrd_dir}_phantom.mhd")
                logging.info(
                    f"Reloading phantom from pickle file {nrrd_dir}")
                # Print the available simulation directories
                sim_dirs = [d for d in os.listdir(
                    os.path.join(data_path, "user_phantoms", nrrd_dir)) if "fastmc_simulation" in d]
                if len(sim_dirs) == 0:
                    logging.info(
                        "No existing simulation directories found. Reload failed")
                    raise ValueError(
                        "No existing simulation directories found. Reload failed")

                if kwargs.get('sim_num', None) is not None:
                    sim_num = ''
                else:
                    sim_num = f"_{kwargs['sim_num']}"
                # Check if the phantom has been pickled
                # Look for a pickle file in this directory
                pickle_files = [f for f in os.listdir(
                    os.path.join(data_path, "user_phantoms",
                                 nrrd_dir, "fastmc_simulation" + '_' + str(kwargs['sim_num']))) if f.endswith('.pkl')]

                phantom = self.load(
                    os.path.join(data_path, "user_phantoms", nrrd_dir, "fastmc_simulation" + '_' + str(kwargs['sim_num']), pickle_files[0]))
                logging.info(
                    f"Phantom loaded from pickle file {pickle_files[0]}")

                self.__dict__.update(phantom.__dict__)

                logging.info("Loading ggems files")

                ggems_output_path = os.path.join(
                    data_path, "user_phantoms", nrrd_dir, "fastmc_output" + '_' + str(kwargs['sim_num']))

                # Check if the ggems output exists
                if not os.path.exists(ggems_output_path):
                    logging.info(
                        "No existing ggems output found. Run a ggems simulation first, can be run with phantom.run_fastmc()")

                ggems_scatter_files = glob.glob(os.path.join(ggems_output_path,  'fastmc_*scatter.mhd')

                                                )
                ggems_primary_files = glob.glob(os.path.join(ggems_output_path,  'fastmc_*[!scatter].mhd')

                                                )

                # Check if the second layer is in the kwargs
                if kwargs.get('second_layer', False):
                    logging.info(
                        "Using the second detector layer ggems files")
                    ggems_scatter_files = [
                        x for x in ggems_scatter_files if 'flood' not in x and '0_2' in x]
                    ggems_primary_files = [
                        x for x in ggems_primary_files if 'flood' not in x and '0_2' in x]
                else:
                    logging.info(
                        "Using the first detector layer ggems files")
                    ggems_scatter_files = [
                        x for x in ggems_scatter_files if 'flood' not in x and '0_2' not in x]
                    ggems_primary_files = [
                        x for x in ggems_primary_files if 'flood' not in x and '0_2' not in x]

                if kwargs.get('second_layer', False):
                    flood_file = os.path.join(
                        ggems_output_path, 'fastmc_00.0_flood_2.mhd')
                else:
                    flood_file = os.path.join(
                        ggems_output_path, 'fastmc_00.0_flood.mhd')
                ggems_scatter_files.sort()
                ggems_primary_files.sort()

                self.load_ggems_files(
                    ggems_scatter_files, ggems_primary_files, flood_file)

                logging.info("ggems files loaded")
                logging.info(
                    "    Scatter files:")
                for f in ggems_scatter_files:
                    logging.info('        ' + f)
                return
            else:
                logging.info(
                    "No existing directory found. Reload failed")
                raise ValueError(
                    "No existing directory found. Reload failed")

        else:
            logging.info(
                f"Found existing directory. Using mhd file {nrrd_dir}_phantom.mhd")

        mhd_file = os.path.join(
            data_path, "user_phantoms",  nrrd_dir, nrrd_dir + "_phantom.mhd")
        range_file = os.path.join(
            data_path, "user_phantoms",  nrrd_dir, nrrd_dir + "_range.txt")
        material_file = os.path.join(
            data_path, "user_phantoms",  nrrd_dir, nrrd_dir + "_materials.txt")
        density_file = os.path.join(
            data_path, "user_phantoms",  nrrd_dir, nrrd_dir + "_density.npy")

        phantom = get_phantom_from_mhd(
            mhd_file, range_file, material_file, is_patient=True)

        self.__dict__.update(phantom.__dict__)
        self.density = np.load(density_file)
        self.density = np.flipud(self.density).T
        self.phan_map = get_phan_map_from_range(
            self.range_file)

        # self.phantom = np.flipud(self.phantom).T
        if nparticles is None:
            raise ValueError(
                "Number of particles must be provided")

        self.nparticles = nparticles

        self.phantom_dir = os.path.join(
            data_path, "user_phantoms", nrrd_dir)

        self.nrrd_file = nrrd_file

        if is_fullfan:
            self.bowtie_file = os.path.join(
                data_path, "bowties", "full_fan_alt.dat")
            self.is_fullfan = True
        else:
            self.bowtie_file = os.path.join(
                data_path, "bowties", "half_fan_mm.dat")
            self.is_fullfan = False

    def load(self, pickle_file, **kwargs):
        '''
        Load the phantom from a pickle file
        '''
        import pickle
        with open(pickle_file, 'rb') as f:
            return pickle.load(f)

    def initialize_fastmc(self, sim_angle, spectrum=None, spec_file=None, **kwargs):
        '''
        Initializes the fastmc simulation

        kwargs: dict
            Dictionary containing the parameters for the fastmc simulation
        '''

        if spec_file is not None:
            self.spectrum_file = spec_file
        elif spectrum is not None:
            spectrum.write_dat_file(os.path.join(
                self.phantom_dir, "spectrum.dat"))
            self.spectrum_file = os.path.join(
                self.phantom_dir, "spectrum.dat")
        else:
            raise ValueError(
                "Spectrum file or spectrum object must be provided")

        # Check if sim_angle is a single value or a list
        if isinstance(sim_angle, int):
            # Making a list of angles
            logging.info("Making a list of angles")
            sim_angles = np.linspace(
                0, np.pi*2, sim_angle, endpoint=False)
            self.sim_angles = sim_angles
        else:
            self.sim_angles = sim_angle

        # Check if detector_params in kwargs
        if 'detector_params' in kwargs:
            for key, value in kwargs['detector_params'].items():
                setattr(self, key, value)
        else:
            self.detector_thickness = 0.6
            self.detector_thickness2 = 0.8
            self.detector_material = 'CsI'

        out_dir = os.path.join(
            self.phantom_dir, "fastmc_output")
        # If the output directory exists then increment the number
        # of the directory
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
            outdir1 = out_dir
        else:
            index = 1
            out_dir1 = out_dir + f"_{index}"
            while os.path.exists(out_dir1):
                out_dir1 = out_dir + f"_{index}"
                index += 1
            os.mkdir(out_dir1)

        self.out_dir = out_dir1

        sim_dir = os.path.join(
            self.phantom_dir, "fastmc_simulation")
        # If the simulation directory exists then increment the number
        # of the directory
        if not os.path.exists(sim_dir):
            os.mkdir(sim_dir)
            sim_dir1 = sim_dir
        else:
            index = 1
            sim_dir1 = sim_dir + f"_{index}"
            while os.path.exists(sim_dir1):
                sim_dir1 = sim_dir + f"_{index}"
                index += 1
            os.mkdir(sim_dir1)

        self.sim_dir = sim_dir1

        self.xx, self.yy = spectrum.get_points()

        write_fastmc_flood_field_xml_file(self, self.sim_dir,
                                          self.out_dir, self.phantom_dir,
                                          half_fan=not self.is_fullfan, file_name=self.sim_dir)

        write_fastmc_xml_file(self, self.sim_dir, self.out_dir, self.phantom_dir,
                              half_fan=not self.is_fullfan)

    def run_fastmc(self, fastmc_path):
        '''
        Run the fastmc simulation

        fastmc_path: str
            The path to the fastmc executable
        '''
        run_fastmc_files(
            lib_path=fastmc_path, sim_dir=self.sim_dir)

    def run_fastcat(self, nphotons, angles, det_on=True, **kwargs):
        '''
        Run the fastcat simulation

        nphotons: int
            The number of photons to be simulated
        angles: int or list
            The number of angles to be simulated
        '''

        # Bowtie in Fastmc is evvery 1 degree angle so we need to interpolate
        # can be linear but that causes some artifacts
        self.bowtie_interp = 'cubic'

        # Check if angles is an int or a list
        if isinstance(angles, int):
            # Making a list of angles
            logging.info("Making a list of angles")
            sim_angles = np.linspace(
                0, np.pi*2, angles, endpoint=False)
        else:
            sim_angles = angles

        # Available pixel pitches
        if self.detector_material == 'CsI':
            pixel_pitches = [100, 150, 336, 392, 784]
        elif self.detector_material == 'GOS':
            pixel_pitches = [336, 392, 784]
        elif self.detector_material == 'CWO':
            pixel_pitches = [261, 392, 784]
        elif self.detector_material == 'CZT':
            pixel_pitches = [330, 342]
        else:
            raise ValueError(
                "Detector material not supported")

        # Find the pixel pitch closest to the detector pixel size
        pixel_size = 1000 * \
            (self.geomet.dDetector[0] +
             self.geomet.dDetector[1])/2
        pixel_pitch_selected = min(
            pixel_pitches, key=lambda x: abs(x-pixel_size))
        self.detector_name = self.detector_material + \
            '-' + str(pixel_pitch_selected) + '-micrometer'

        logging.info(
            f"Detector matching specifications: {self.detector_name}")

        spectrum = Spectrum()
        spectrum.x = self.xx
        spectrum.y = self.yy

        det = Detector(spectrum, self.detector_name)

        self.nphotons_sim = nphotons
        self.simulate(det, spectrum, sim_angles, nphoton=self.nphotons_sim, mgy=0,
                      ASG=False, scat_on=False,
                      det_on=det_on,
                      bowtie=True,
                      return_intensity=True)

    def analyse_515(self, slc, place, fmt="-"):
        pass

    # Define a custom print function
    def __str__(self):
        # Print a summary of the phantom
        information = f'''
Phantom Summary:
----------------

    Name: {self.nrrd_file}
    Phantom dir: {self.phantom_dir}

    Detector material: {self.detector_material if hasattr(self, 'detector_material') else None}
    Detector thickness: {self.detector_thickness if hasattr(self, 'detector_thickness') else None}

    Bowtie file: {self.bowtie_file}
    Material file: {self.material_file}
    MHD file: {self.mhd_file}
    Range file: {self.range_file}

    Number of particles: {self.nparticles}
    Is full fan: {self.is_fullfan}

Geometry: 
    
{self.geomet}
        '''
        return information
