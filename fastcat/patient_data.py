#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
Fastcat a rapid and accurate CBCT simulator.

PHANTOMS: CODE CONTAINING PHANTOM PARAMETERS AND ANALYSIS

"""

from __future__ import print_function

import logging
import os

import numpy as np
from numpy import cos, sin

from fastcat.ggems_simulate import Phantom
from fastcat.utils import get_phantom_from_mhd, nrrd_to_mhd, get_phan_map_from_range
from fastcat.fastmc_scatter import write_fastmc_xml_file, run_fastmc_files, write_fastmc_flood_field_xml_file
from fastcat.detector import Detector

data_path = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "data")


class patient_phantom(Phantom):
    def __init__(self, nrrd_file, nparticles, is_fullfan=True, geo=None):
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
        '''
        nrrd_dir = nrrd_file.split('/')[-1].split(".")[0]

        if not os.path.exists(os.path.join(data_path, "user_phantoms", nrrd_dir)):
            logging.info(
                "No existing directory found. Converting nrrd to mhd")
            nrrd_to_mhd(nrrd_file)
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
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        self.out_dir = out_dir

        sim_dir = os.path.join(
            self.phantom_dir, "fastmc_simulation")
        if not os.path.exists(sim_dir):
            os.mkdir(sim_dir)
        self.sim_dir = sim_dir

        self.spectrum = spectrum

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
        det = Detector(self.spectrum, self.detector_name)

        self.nphotons_sim = nphotons
        self.simulate(det, self.spectrum, sim_angles, nphoton=self.nphotons_sim, mgy=0,
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
