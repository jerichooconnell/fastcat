#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
Fastcat a rapid and accurate CBCT simulator.

PHANTOMS: CODE CONTAINING PHANTOM PARAMETERS AND ANALYSIS

"""

from __future__ import print_function

import logging
import os

import astropy.stats as stats
import numpy as np
import tigre
from numpy import cos, sin

from scipy.signal import filtfilt, butter, find_peaks
from fastcat.simulate import Phantom

data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


class Catphan_515(Phantom):
    """
    The subslice contrast module from a Catphan
    """

    def __init__(self):  # ,det):
        self.phantom = np.load(
            os.path.join(
                data_path, "phantoms", "catphan_low_contrast_512_8cm.npy"
            )
        )  # 'catphan_low_contrast_512_8cm.npy')) # Paper 2 uses the 8cm btw
        self.geomet = tigre.geometry_default(nVoxel=self.phantom.shape)
        self.geomet.DSD = 1520  # 1500 + 20 for det casing
        self.geomet.nDetector = np.array([64, 512])
        self.geomet.dDetector = np.array(
            [0.784, 0.784]
        )  # det.pitch, det.pitch]) #TODO: Change this to get phantom

        # I think I can get away with this
        self.geomet.sDetector = self.geomet.dDetector * self.geomet.nDetector

        self.geomet.sVoxel = np.array((160, 160, 160))
        self.geomet.dVoxel = self.geomet.sVoxel / self.geomet.nVoxel
        self.phan_map = [
            "air",
            "water",
            "G4_LUNG_ICRP",
            "G4_BONE_COMPACT_ICRU",
            "G4_BONE_CORTICAL_ICRP",
            "G4_ADIPOSE_TISSUE_ICRP",
            "G4_BRAIN_ICRP",
            "G4_B-100_BONE",
        ]

    def analyse_515(self, recon_slice, place=None, run_name=""):
        def create_mask(shape):

            im = np.zeros(shape)
            ii = 1

            offset = 0  # 0.1

            first_radius = 5 - offset
            second_radius = 2.5 - offset

            correction = 0  # -2*np.pi/180
            # CTMAT(x) formel=H2O dichte=x
            #             LEN = 100

            A0 = 87.7082 * np.pi / 180 + correction
            A1 = 108.3346 * np.pi / 180 + correction
            A2 = 126.6693 * np.pi / 180 + correction
            A3 = 142.7121 * np.pi / 180 + correction
            A4 = 156.4631 * np.pi / 180 + correction
            A5 = 167.9223 * np.pi / 180 + correction
            A6 = 177.0896 * np.pi / 180 + correction
            A7 = 183.9651 * np.pi / 180 + correction
            A8 = 188.5487 * np.pi / 180 + correction

            B0 = 110.6265 * np.pi / 180 + correction
            B1 = 142.7121 * np.pi / 180 + correction
            B2 = 165.6304 * np.pi / 180 + correction
            B3 = 179.3814 * np.pi / 180 + correction

            tad = 0.2

            # Phantom
            # ++++ module body ++++++++++++++++++++ */
            create_circular_mask(
                x=0.000, y=0.000, r=-tad + 2, index=ii, image=im
            )

            ii += 1

            # ++++ supra-slice 1.0% targets ++++++ */
            create_circular_mask(
                x=first_radius * cos(A0),
                y=first_radius * sin(A0),
                r=-tad + 0.75,
                index=ii,
                image=im,
            )
            ii += 1
            create_circular_mask(
                x=first_radius * cos(A1),
                y=first_radius * sin(A1),
                r=-tad + 0.45,
                index=ii,
                image=im,
            )
            ii += 1
            create_circular_mask(
                x=first_radius * cos(A2),
                y=first_radius * sin(A2),
                r=-tad + 0.40,
                index=ii,
                image=im,
            )
            ii += 1
            create_circular_mask(
                x=first_radius * cos(A3),
                y=first_radius * sin(A3),
                r=-tad + 0.35,
                index=ii,
                image=im,
            )
            ii += 1
            create_circular_mask(
                x=first_radius * cos(A4),
                y=first_radius * sin(A4),
                r=-tad + 0.30,
                index=ii,
                image=im,
            )
            ii += 1
            create_circular_mask(
                x=first_radius * cos(A5),
                y=first_radius * sin(A5),
                r=-tad + 0.25,
                index=ii,
                image=im,
            )
            ii += 1
            create_circular_mask(
                x=first_radius * cos(A6),
                y=first_radius * sin(A6),
                r=-tad + 0.20,
                index=ii,
                image=im,
            )
            ii += 1
            create_circular_mask(
                x=first_radius * cos(A7),
                y=first_radius * sin(A7),
                r=-tad + 0.15,
                index=ii,
                image=im,
            )
            ii += 1
            create_circular_mask(
                x=first_radius * cos(A8),
                y=first_radius * sin(A8),
                r=-tad + 0.10,
                index=ii,
                image=im,
            )
            ii += 1

            # ++++ supra-slice 0.3% targets +++++++++++++++++++++ */
            create_circular_mask(
                x=first_radius * cos(A0 + 2 / 3 * np.pi),
                y=first_radius * sin(A0 + 2 / 3 * np.pi),
                r=-tad + 0.75,
                index=ii,
                image=im,
            )
            ii += 1
            create_circular_mask(
                x=first_radius * cos(A1 + 2 / 3 * np.pi),
                y=first_radius * sin(A1 + 2 / 3 * np.pi),
                r=-tad + 0.45,
                index=ii,
                image=im,
            )
            ii += 1
            create_circular_mask(
                x=first_radius * cos(A2 + 2 / 3 * np.pi),
                y=first_radius * sin(A2 + 2 / 3 * np.pi),
                r=-tad + 0.40,
                index=ii,
                image=im,
            )
            ii += 1
            create_circular_mask(
                x=first_radius * cos(A3 + 2 / 3 * np.pi),
                y=first_radius * sin(A3 + 2 / 3 * np.pi),
                r=-tad + 0.35,
                index=ii,
                image=im,
            )
            ii += 1
            create_circular_mask(
                x=first_radius * cos(A4 + 2 / 3 * np.pi),
                y=first_radius * sin(A4 + 2 / 3 * np.pi),
                r=-tad + 0.30,
                index=ii,
                image=im,
            )
            ii += 1
            create_circular_mask(
                x=first_radius * cos(A5 + 2 / 3 * np.pi),
                y=first_radius * sin(A5 + 2 / 3 * np.pi),
                r=-tad + 0.25,
                index=ii,
                image=im,
            )
            ii += 1
            create_circular_mask(
                x=first_radius * cos(A6 + 2 / 3 * np.pi),
                y=first_radius * sin(A6 + 2 / 3 * np.pi),
                r=-tad + 0.20,
                index=ii,
                image=im,
            )
            ii += 1
            create_circular_mask(
                x=first_radius * cos(A7 + 2 / 3 * np.pi),
                y=first_radius * sin(A7 + 2 / 3 * np.pi),
                r=-tad + 0.15,
                index=ii,
                image=im,
            )
            ii += 1
            create_circular_mask(
                x=first_radius * cos(A8 + 2 / 3 * np.pi),
                y=first_radius * sin(A8 + 2 / 3 * np.pi),
                r=-tad + 0.10,
                index=ii,
                image=im,
            )
            ii += 1

            # ++++ supra-slice 0.5% targets +++++++++++++++++++++++++ */
            create_circular_mask(
                x=first_radius * cos(A0 + 4 / 3 * np.pi),
                y=first_radius * sin(A0 + 4 / 3 * np.pi),
                r=-tad + 0.75,
                index=ii,
                image=im,
            )
            ii += 1
            create_circular_mask(
                x=first_radius * cos(A1 + 4 / 3 * np.pi),
                y=first_radius * sin(A1 + 4 / 3 * np.pi),
                r=-tad + 0.45,
                index=ii,
                image=im,
            )
            ii += 1
            create_circular_mask(
                x=first_radius * cos(A2 + 4 / 3 * np.pi),
                y=first_radius * sin(A2 + 4 / 3 * np.pi),
                r=-tad + 0.40,
                index=ii,
                image=im,
            )
            ii += 1
            create_circular_mask(
                x=first_radius * cos(A3 + 4 / 3 * np.pi),
                y=first_radius * sin(A3 + 4 / 3 * np.pi),
                r=-tad + 0.35,
                index=ii,
                image=im,
            )
            ii += 1
            create_circular_mask(
                x=first_radius * cos(A4 + 4 / 3 * np.pi),
                y=first_radius * sin(A4 + 4 / 3 * np.pi),
                r=-tad + 0.30,
                index=ii,
                image=im,
            )
            ii += 1
            create_circular_mask(
                x=first_radius * cos(A5 + 4 / 3 * np.pi),
                y=first_radius * sin(A5 + 4 / 3 * np.pi),
                r=-tad + 0.25,
                index=ii,
                image=im,
            )
            ii += 1
            create_circular_mask(
                x=first_radius * cos(A6 + 4 / 3 * np.pi),
                y=first_radius * sin(A6 + 4 / 3 * np.pi),
                r=-tad + 0.20,
                index=ii,
                image=im,
            )
            ii += 1
            create_circular_mask(
                x=first_radius * cos(A7 + 4 / 3 * np.pi),
                y=first_radius * sin(A7 + 4 / 3 * np.pi),
                r=-tad + 0.15,
                index=ii,
                image=im,
            )
            ii += 1
            create_circular_mask(
                x=first_radius * cos(A8 + 4 / 3 * np.pi),
                y=first_radius * sin(A8 + 4 / 3 * np.pi),
                r=-tad + 0.10,
                index=ii,
                image=im,
            )
            ii += 1

            # ++++ subslice 1.0% targets 7mm long  */
            create_circular_mask(
                x=second_radius * cos(B0),
                y=second_radius * sin(B0),
                r=-tad + 0.45,
                index=ii,
                image=im,
            )
            ii += 1
            create_circular_mask(
                x=second_radius * cos(B1),
                y=second_radius * sin(B1),
                r=-tad + 0.35,
                index=ii,
                image=im,
            )
            ii += 1
            create_circular_mask(
                x=second_radius * cos(B2),
                y=second_radius * sin(B2),
                r=-tad + 0.25,
                index=ii,
                image=im,
            )
            ii += 1
            create_circular_mask(
                x=second_radius * cos(B3),
                y=second_radius * sin(B3),
                r=-tad + 0.15,
                index=ii,
                image=im,
            )
            ii += 1

            # ++++ subslice 1.0% targets 3mm long  */
            create_circular_mask(
                x=second_radius * cos(B0 + 2 / 3 * np.pi),
                y=second_radius * sin(B0 + 2 / 3 * np.pi),
                r=-tad + 0.45,
                index=ii,
                image=im,
            )
            ii += 1
            create_circular_mask(
                x=second_radius * cos(B1 + 2 / 3 * np.pi),
                y=second_radius * sin(B1 + 2 / 3 * np.pi),
                r=-tad + 0.35,
                index=ii,
                image=im,
            )
            ii += 1
            create_circular_mask(
                x=second_radius * cos(B2 + 2 / 3 * np.pi),
                y=second_radius * sin(B2 + 2 / 3 * np.pi),
                r=-tad + 0.25,
                index=ii,
                image=im,
            )
            ii += 1
            create_circular_mask(
                x=second_radius * cos(B3 + 2 / 3 * np.pi),
                y=second_radius * sin(B3 + 2 / 3 * np.pi),
                r=-tad + 0.15,
                index=ii,
                image=im,
            )
            ii += 1

            # ++++ subslice 1.0% targets 5mm long  */
            create_circular_mask(
                x=second_radius * cos(B0 + 4 / 3 * np.pi),
                y=second_radius * sin(B0 + 4 / 3 * np.pi),
                r=-tad + 0.45,
                index=ii,
                image=im,
            )
            ii += 1
            create_circular_mask(
                x=second_radius * cos(B1 + 4 / 3 * np.pi),
                y=second_radius * sin(B1 + 4 / 3 * np.pi),
                r=-tad + 0.35,
                index=ii,
                image=im,
            )
            ii += 1
            create_circular_mask(
                x=second_radius * cos(B2 + 4 / 3 * np.pi),
                y=second_radius * sin(B2 + 4 / 3 * np.pi),
                r=-tad + 0.25,
                index=ii,
                image=im,
            )
            ii += 1
            create_circular_mask(
                x=second_radius * cos(B3 + 4 / 3 * np.pi),
                y=second_radius * sin(B3 + 4 / 3 * np.pi),
                r=-tad + 0.15,
                index=ii,
                image=im,
            )
            ii += 1

            return im

        def create_mask_multi(shape):

            im = np.zeros(shape)
            ii = 1

            correction = 0
            first_radius = 5
            second_radius = 2.5

            # CTMAT(x) formel=H2O dichte=x

            A0 = 87.7082 * np.pi / 180 + correction
            B0 = 110.6265 * np.pi / 180 + correction

            # Phantom
            # ++++ module body  */
            create_circular_mask(x=0.000, y=0.000, r=1.0, index=ii, image=im)

            ii += 1

            tad = 0.5
            # ++++ supra-slice 1.0% targets  */

            create_circular_mask(
                x=first_radius * cos(A0),
                y=first_radius * sin(A0),
                r=0.75 - tad,
                index=ii,
                image=im,
            )
            ii += 1
            create_circular_mask(
                x=first_radius * cos(A0 + 2 / 3 * np.pi),
                y=first_radius * sin(A0 + 2 / 3 * np.pi),
                r=0.75 - tad,
                index=ii,
                image=im,
            )
            ii += 1
            create_circular_mask(
                x=first_radius * cos(A0 + 4 / 3 * np.pi),
                y=first_radius * sin(A0 + 4 / 3 * np.pi),
                r=0.75 - tad,
                index=ii,
                image=im,
            )
            ii += 1

            create_circular_mask(
                x=second_radius * cos(B0),
                y=second_radius * sin(B0),
                r=0.45 - tad,
                index=ii,
                image=im,
            )
            ii += 1
            create_circular_mask(
                x=second_radius * cos(B0 + 2 / 3 * np.pi),
                y=second_radius * sin(B0 + 2 / 3 * np.pi),
                r=0.45 - tad,
                index=ii,
                image=im,
            )
            ii += 1
            create_circular_mask(
                x=second_radius * cos(B0 + 4 / 3 * np.pi),
                y=second_radius * sin(B0 + 4 / 3 * np.pi),
                r=0.45 - tad,
                index=ii,
                image=im,
            )
            ii += 1
            return im

        def create_circular_mask(x, y, r, index, image):

            h, w = image.shape

            center = [
                x * int(w / 2) / 8 + int(w / 2),
                y * int(h / 2) / 8 + int(h / 2),
            ]

            if center is None:  # use the middle of the image
                center = (int(w / 2), int(h / 2))
            # if (
            #     r is None
            # ):  # use the smallest distance between
            # the center and image walls

            Y, X = np.ogrid[:h, :w]
            dist_from_center = np.sqrt(
                (X - center[0]) ** 2 + (Y - center[1]) ** 2
            )

            mask = dist_from_center <= r * int(w / 2) / 8

            image[mask] = index

        im = create_mask(recon_slice.shape)

        contrast = []
        noise = []
        cnr = []
        ci = []

        ii = 1

        ref_mean = np.mean(recon_slice[im == ii])
        # ref_std = np.std(recon_slice[im == ii])

        for ii in range(2, int(np.max(im) + 1)):

            nsample = len(recon_slice[im == ii])

            if nsample > 2:

                noise.append(np.std(recon_slice[im == ii]))

                booted = np.abs(
                    stats.bootstrap(
                        recon_slice[im == ii],
                        100,
                        samples=int(nsample / 5),
                        bootfunc=np.mean,
                    )
                    - ref_mean
                )

                ci.append(np.std(booted))
                contrast.append(np.mean(booted))

                cnr.append(contrast[-1] / (np.sqrt(noise[-1] ** 2)))

        ci_v = [2 * (ci[ii] / ref_mean) * 100 for ii in range(len(ci))]

        rs = np.linspace(0.1, 0.45, 8)

        inds_i_want = [0, 6, 12, 18, 21, 24]
        ww = 0.085

        shorts = [
            "Lung",
            "Compact Bone",
            "Cortical Bone",
            "Adipose",
            "Brain",
            "B-100",
        ]

        contrasts_i_want = np.array(
            [(contrast[ii] / ref_mean) * 100 for ii in range(len(contrast))]
        )[inds_i_want]

        place[0].errorbar(
            np.arange(len(inds_i_want)),
            contrasts_i_want,
            capsize=ww + 1.5,
            yerr=np.array(ci_v)[inds_i_want],
            fmt="x",
            label=run_name,
        )
        place[0].set_xticks(range(len(inds_i_want)))
        place[0].set_xticklabels(shorts, fontsize=12, rotation=75)
        place[0].set_ylabel("% Contrast")
        place[0].set_title("Contrast")
        place[0].legend()

        place[1].errorbar(
            np.arange(len(inds_i_want)),
            np.array(cnr)[inds_i_want],
            capsize=ww + 1.5,
            yerr=np.array(cnr)[inds_i_want]
            * (np.array(ci_v)[inds_i_want] / contrasts_i_want),
            fmt="x",
        )
        place[1].set_xticks(range(len(inds_i_want)))
        place[1].set_xticklabels(shorts, fontsize=12, rotation=75)
        place[1].set_ylabel("CNR")
        place[1].set_title("Contrast to Noise")

        return_contrast = True

        if return_contrast:
            return (
                rs,
                [
                    (contrast[ii] / ref_mean) * 100
                    for ii in range(len(contrast))
                ],
                ci_v,
                cnr,
                np.array(cnr)[inds_i_want]
                * (np.array(ci_v)[inds_i_want] / contrasts_i_want),
            )


class Catphan_404(Phantom):
    def __init__(self, pitch=0.784, hi_res=True):  # ,det):
        if hi_res:
            self.phantom = np.load(
                os.path.join(
                    data_path,
                    "phantoms",
                    "catphan_sensiometry_512_10cm_mod.npy",
                )
            )  # 10cm.npy'))
        else:
            self.phantom = np.load(
                os.path.join(
                    data_path, "phantoms", "catphan_sensiometry_512_8cm.npy"
                )
            )  # 10cm.npy'))
            logging.info("Phantom is low resolution")
        # The 10cm is really the 8cm equivalent
        self.geomet = tigre.geometry_default(nVoxel=self.phantom.shape)
        self.geomet.DSO = 1000
        self.geomet.DSD = 1500  # 1520 JO dec 2020 1500 + 20 for det casing
        self.geomet.nDetector = np.array([64, 512])
        self.geomet.dDetector = np.array(
            [pitch, pitch]
        )  # det.pitch, det.pitch]) #TODO: Change this to get phantom

        # I think I can get away with this
        self.geomet.sDetector = self.geomet.dDetector * self.geomet.nDetector
        self.geomet.sVoxel = np.array((160, 200, 200))
        self.geomet.dVoxel = self.geomet.sVoxel / self.geomet.nVoxel

        self.phan_map = [
            "air",
            "G4_POLYSTYRENE",
            "G4_POLYVINYL_BUTYRAL",
            "G4_POLYVINYL_BUTYRAL",
            "CATPHAN_Delrin",
            "G4_POLYVINYL_BUTYRAL",
            "CATPHAN_Teflon_revised",
            "air",
            "CATPHAN_PMP",
            "G4_POLYVINYL_BUTYRAL",
            "CATPHAN_LDPE",
            "G4_POLYVINYL_BUTYRAL",
            "CATPHAN_Polystyrene",
            "air",
            "CATPHAN_Acrylic",
            "air",
            "CATPHAN_Teflon",
            "air",
            "air",
            "air",
            "air",
        ]

    def analyse_515(self, recon_slice, place=None, run_name=""):
        def create_mask(shape):

            im = np.zeros(shape)
            ii = 1

            # offset = 0.1

            # first_radius = 5

            A0 = 90.0 * np.pi / 180

            # Phantom
            # ++++ module body  */
            create_circular_mask(
                x=6 * cos(A0 + 1 / 4 * np.pi),
                y=6 * sin(A0 + 1 / 4 * np.pi),
                r=0.5,
                index=ii,
                image=im,
            )

            ii += 1

            # ++++ supra-slice 1.0% targets  */

            # ++++ supra-slice 1.0% targets  */
            create_circular_mask(
                x=6 * cos(A0), y=6 * sin(A0), r=0.5, index=ii, image=im
            )

            ii += 1

            # ++++ supra-slice 0.3% targets  */
            create_circular_mask(
                x=6 * cos(A0 + 1 / 2 * np.pi),
                y=6 * sin(A0 + 1 / 2 * np.pi),
                r=0.5,
                index=ii,
                image=im,
            )

            ii += 1

            # ++++ supra-slice 0.5% targets  */
            create_circular_mask(
                x=6 * cos(A0 + np.pi),
                y=6 * sin(A0 + np.pi),
                r=0.5,
                index=ii,
                image=im,
            )

            ii += 1

            # ++++ supra-slice 0.5% targets  */
            create_circular_mask(
                x=6 * cos(A0 + 3 / 2 * np.pi),
                y=6 * sin(A0 + 3 / 2 * np.pi),
                r=0.5,
                index=ii,
                image=im,
            )

            return im

        def create_circular_mask(x, y, r, index, image):

            h, w = image.shape

            center = [
                x * int(w / 2) / 8 + int(w / 2),
                y * int(h / 2) / 8 + int(h / 2),
            ]

            if center is None:  # use the middle of the image
                center = (int(w / 2), int(h / 2))
            # if (
            #     r is None
            # ):  # use the smallest distance
            # between the center and image walls

            Y, X = np.ogrid[:h, :w]
            dist_from_center = np.sqrt(
                (X - center[0]) ** 2 + (Y - center[1]) ** 2
            )

            mask = dist_from_center <= r * int(w / 2) / 8

            image[mask] = index

        im = create_mask(recon_slice.shape)

        contrast = []
        noise = []
        cnr = []
        ci = []

        ii = 1

        ref_mean = np.mean(recon_slice[im == ii])
        # ref_std = np.std(recon_slice[im == ii])

        for ii in range(2, int(np.max(im) + 1)):

            nsample = len(recon_slice[im == ii])

            if nsample > 2:

                noise.append(np.std(recon_slice[im == ii]))

                booted = np.abs(
                    stats.bootstrap(
                        recon_slice[im == ii],
                        100,
                        samples=int(nsample / 5),
                        bootfunc=np.mean,
                    )
                    - ref_mean
                )

                ci.append(np.std(booted))
                contrast.append(np.mean(booted))

                cnr.append(contrast[-1] / (np.sqrt(noise[-1] ** 2)))

        ci_v = [2 * (ci[ii] / ref_mean) * 100 for ii in range(len(ci))]

        rs = np.linspace(0.1, 0.45, 8)

        inds_i_want = [0, 1, 2, 3]
        ww = 0.085

        shorts = ["Lung", "Compact Bone", "Cortical Bone", "Adipose"]

        contrasts_i_want = np.array(
            [(contrast[ii] / ref_mean) * 100 for ii in range(len(contrast))]
        )[inds_i_want]

        place[0].errorbar(
            np.arange(len(inds_i_want)),
            contrasts_i_want,
            capsize=ww + 1.5,
            yerr=np.array(ci_v)[inds_i_want],
            fmt="x",
            label=run_name,
        )
        place[0].set_xticks(range(len(inds_i_want)))
        place[0].set_xticklabels(shorts, fontsize=12, rotation=75)
        place[0].set_ylabel("% Contrast")
        place[0].set_title("Contrast")
        place[0].legend()

        place[1].errorbar(
            np.arange(len(inds_i_want)),
            np.array(cnr)[inds_i_want],
            capsize=ww + 1.5,
            yerr=np.array(cnr)[inds_i_want]
            * (np.array(ci_v)[inds_i_want] / contrasts_i_want),
            fmt="x",
        )
        place[1].set_xticks(range(len(inds_i_want)))
        place[1].set_xticklabels(
            shorts[: len(inds_i_want)], fontsize=12, rotation=75
        )
        place[1].set_ylabel("CNR")
        place[1].set_title("Contrast to Noise")

        return_contrast = True

        if return_contrast:
            return (
                rs,
                [
                    (contrast[ii] / ref_mean) * 100
                    for ii in range(len(contrast))
                ],
                ci_v,
                cnr,
                np.array(cnr)[inds_i_want]
                * (np.array(ci_v)[inds_i_want] / contrasts_i_want),
            )

class Catphan_404_Devon(Phantom):
    def __init__(self, pitch=0.33, hi_res=True):
        if hi_res:
            self.phantom = np.load(
                os.path.join(
                    data_path,
                    "phantoms",
                    "catphan_sensiometry_512_10cm_mod.npy",
                )
            )  # 10cm.npy'))
        else:
            self.phantom = np.load(
                os.path.join(
                    data_path, "phantoms", "catphan_sensiometry_512_8cm.npy"
                )
            )
            logging.info("Phantom is low resolution")
        
        self.scatter = 'devon_total.npy'
        self.scatter_coords = np.linspace(-288* 0.033 - 0.0165, 288 * 0.033 -0.165, 576)
        # The 10cm is really the 8cm equivalent
        self.geomet = tigre.geometry_default(nVoxel=self.phantom.shape)
        self.geomet.DSO = 322
        self.geomet.DSD = 322 + 266  # 1520 JO dec 2020 1500 + 20 for det casing
        self.geomet.nDetector = np.array([64, 576])
        self.geomet.dDetector = np.array(
            [pitch, pitch]
        )
        # I think I can get away with this
        self.geomet.sDetector = self.geomet.dDetector * self.geomet.nDetector
        self.geomet.sVoxel = np.array((50, 100, 100))
        self.geomet.dVoxel = self.geomet.sVoxel / self.geomet.nVoxel

        self.phan_map = [
            "air",
            "G4_POLYSTYRENE",
            "G4_POLYVINYL_BUTYRAL",
            "G4_POLYVINYL_BUTYRAL",
            "CATPHAN_Delrin",
            "G4_POLYVINYL_BUTYRAL",
            "CATPHAN_Teflon_revised",
            "air",
            "CATPHAN_PMP",
            "G4_POLYVINYL_BUTYRAL",
            "CATPHAN_LDPE",
            "G4_POLYVINYL_BUTYRAL",
            "CATPHAN_Polystyrene",
            "air",
            "CATPHAN_Acrylic",
            "air",
            "CATPHAN_Teflon",
            "air",
            "air",
            "air",
            "air",
        ]

    def analyse_515(self, recon_slice, place=None, run_name=""):
        pass
    
class Catphan_MTF(Phantom):
    def __init__(self):
        self.phantom = np.load(
            os.path.join(data_path, "phantoms", "MTF_phantom_1024.npy")
        )
        self.geomet = tigre.geometry_default(nVoxel=self.phantom.shape)
        self.geomet.nDetector = np.array([64, 512])
        self.geomet.dDetector = np.array([0.784, 0.784])
        self.phan_map = ["air", "water", "G4_BONE_COMPACT_ICRU"]
        self.geomet.DSD = 1500
        # I think I can get away with this
        self.geomet.sDetector = self.geomet.dDetector * self.geomet.nDetector

        self.geomet.sVoxel = np.array((160, 160, 160))
        self.geomet.dVoxel = self.geomet.sVoxel / self.geomet.nVoxel

    def analyse_515(self, slc, place, fmt="-"):

        chunk_x = 100
        chunk_y = 35

        signal = []
        standev = []

        def get_diff(smoothed_slice, h0, d0):

            peak_info = find_peaks(
                smoothed_slice, height=h0 / 3, prominence=d0 / 5
            )

            peaks = peak_info[0]
            valleys = np.unique(
                np.hstack(
                    [peak_info[1]["right_bases"], peak_info[1]["left_bases"]]
                )
            )

            if len(valleys) > 0:
                inds = np.array(valleys < np.max(peaks)) * np.array(
                    valleys > np.min(peaks)
                )
            else:
                return

            valleys = valleys[inds]
            peaks = peaks[:-1]

            diffs = []

            for peak, valley in zip(peaks, valleys):

                diff = smoothed_slice[peak] - smoothed_slice[valley]

                diffs.append(diff)

            return diffs

        start_x = 310
        start_y = 270

        b, a = butter(3, 1 / 7, btype="low", analog=False)

        # smoothed_slice = filtfilt(
        # b, a, np.mean(slc[start_y:start_y+chunk_y,
        # start_x:start_x+chunk_x],0))
        # #np.convolve(np.mean(slc[start_
        # y:start_y+chunk_y,start_x:start_x+chunk_x],0),10*[0.1],'same')

        high = np.max(np.mean(slc[270:290, 330:340], 1))
        low = np.min(np.mean(slc[270:290, 360:370], 1))

        signal.append(high - low)
        standev.append(0)

        b, a = butter(3, 1 / 5, btype="low", analog=False)
        b2, a2 = butter(3, 1 / 3.5, btype="low", analog=False)
        b3, a3 = butter(3, 1 / 2, btype="low", analog=False)

        for start_y, freq in zip(
            [400, 520, 650], [[b, a], [b2, a2], [b3, a3]]
        ):
            for start_x in [690, 570, 430, 300]:

                # Need this to correct against the artifacts
                # if start_y == 400:
                # b, a = butter(3, 1/freq, btype='low', analog=False)
                smoothed_slice = filtfilt(
                    freq[0],
                    freq[1],
                    np.mean(
                        slc[
                            start_y : start_y + chunk_y,
                            start_x : start_x + chunk_x,
                        ],
                        0,
                    ),
                )
                # else:
                #     smoothed_slice =
                # np.mean(slc[start_y:start_y+chunk_y,start_x:start_x+chunk_x],0)

                diffs = get_diff(smoothed_slice, high, signal[0])

                if diffs is None:
                    signal.append(0)
                    standev.append(0)
                    break

                if len(diffs) != 0:
                    signal.append(np.mean(diffs))
                    standev.append(np.std(diffs))
                else:
                    signal.append(0)
                    standev.append(0)
                    break

        pitch = 0.015625

        lpmm = [1 / (2 * ii * pitch) for ii in range(12, 0, -1)]

        lpmm.insert(0, 1 / (2 * 30 * pitch))

        place[0].errorbar(
            lpmm[: len(signal)],
            signal / signal[0],
            yerr=standev / signal[0],
            fmt="kx",
        )
        place[0].plot(lpmm[: len(signal)], signal / signal[0], fmt)
        place[0].set_xlabel("lp/cm")
        place[0].set_ylabel("MTF")

        return [signal / signal[0], standev / signal[0]]


class Catphan_projections(Phantom):
    def __init__(self):
        self.phantom = np.load(
            os.path.join(
                data_path, "phantoms", "catphan_projection_512_10cm.npy"
            )
        ).T
        self.geomet = tigre.geometry_default(nVoxel=self.phantom.shape)
        self.geomet.nDetector = np.array([512, 512])
        self.geomet.dDetector = np.array([0.784, 0.784])
        self.phan_map = [
            "air",
            "water",
            "CB2-30",
            "adipose",
            "water",
            "water",
            "G4_LUNG_ICRP",
            "tissue4",
            "testis",
            "brain",
            "tissue",
            "tissue4",
            "testis",
            "brain",
            "breast",
            "muscle",
            "G4_MUSCLE_SKELETAL_ICRP",
            "G4_MUSCLE_STRIATED_ICRU",
            "G4_SKIN_ICRP",
            "G4_TISSUE-PROPANE",
            "G4_TISSUE-METHANE",
            "G4_TISSUE_SOFT_ICRP",
            "G4_TISSUE_SOFT_ICRU-4",
            "G4_BLOOD_ICRP",
            "G4_BODY",
            "G4_BONE_COMPACT_ICRU",
            "G4_BONE_CORTICAL_ICRP",
        ]

        self.geomet.DSD = 1500
        # I think I can get away with this
        self.geomet.sDetector = self.geomet.dDetector * self.geomet.nDetector

        self.geomet.sVoxel = np.array((160, 160, 150))
        self.geomet.dVoxel = self.geomet.sVoxel / self.geomet.nVoxel

    def analyse_515(self, slc, place=None, fmt="-"):

        CNR = np.abs(
            np.mean(slc[250:260, 250:260]) - np.mean(slc[310:320, 250:260])
        ) / np.sqrt(
            np.std(slc[250:260, 250:260]) ** 2
            + np.std(slc[310:320, 250:260]) ** 2
        )

        return CNR


class XCAT(Phantom):
    def __init__(self):
        self.phantom = np.load(
            os.path.join(data_path, "phantoms", "ct_scan_smaller.npy")
        )
        self.geomet = tigre.geometry_default(nVoxel=self.phantom.shape)
        self.geomet.nDetector = np.array([512, 512])
        self.geomet.dDetector = np.array([0.784, 0.784])
        self.phan_map = [
            "air",
            "lung",
            "adipose",
            "adipose",
            "water",
            "water",
            "G4_LUNG_ICRP",
            "tissue4",
            "testis",
            "brain",
            "tissue",
            "tissue4",
            "testis",
            "brain",
            "breast",
            "muscle",
            "G4_MUSCLE_SKELETAL_ICRP",
            "G4_MUSCLE_STRIATED_ICRU",
            "G4_SKIN_ICRP",
            "G4_TISSUE-PROPANE",
            "G4_TISSUE-METHANE",
            "G4_TISSUE_SOFT_ICRP",
            "G4_TISSUE_SOFT_ICRU-4",
            "G4_BLOOD_ICRP",
            "G4_BODY",
            "G4_BONE_COMPACT_ICRU",
            "G4_BONE_CORTICAL_ICRP",
        ]

        self.geomet.DSD = 1500
        # I think I can get away with this
        self.geomet.sDetector = self.geomet.dDetector * self.geomet.nDetector

        self.geomet.sVoxel = np.array((160, 160, 160))
        self.geomet.dVoxel = self.geomet.sVoxel / self.geomet.nVoxel

    def analyse_515(self, slc, place, fmt="-"):

        pass


class XCAT2(Phantom):
    def __init__(self):

        head = True
        if head:
            self.phantom = np.load(
                os.path.join(data_path, "phantoms", "ct_scan_head_small.npy")
            )
        else:
            self.phantom = np.load(
                os.path.join(data_path, "phantoms", "ct_scan_smaller.npy")
            )
        self.geomet = tigre.geometry_default(nVoxel=self.phantom.shape)
        self.geomet.nDetector = np.array([124, 512])
        self.geomet.dDetector = np.array([0.784, 0.784])

        if head:
            self.phan_map = [
                "air",
                "G4_LUNG_LD_ICRP",
                "G4_ADIPOSE_TISSUE_ICRP2",
                "water",
                "RED_MARROW_ICRP",
                "G4_BRAIN_ICRP",
                "G4_MUSCLE_SKELETAL_ICRP",
                "THYROID_ICRP",
                "blood",
                "G4_EYE_LENS_ICRP",
                "CARTILAGE_ICRP",
                "C4_Vertebra_ICRP",
                "SKULL_ICRP",
            ]
        else:
            self.phan_map = [
                "air",
                "air",
                "G4_LUNG_LD_ICRP",
                "G4_ADIPOSE_TISSUE_ICRP2",
                "water",
                "RED_MARROW_ICRP",
                "INTESTINE_ICRP",
                "PANCREAS_ICRP",
                "G4_MUSCLE_SKELETAL_ICRP",
                "KIDNEY_ICRP",
                "HEART_ICRP",
                "THYROID_ICRP",
                "LIVER_ICRP",
                "blood",
                "SPLEEN_ICRP",
                "CARTILAGE_ICRP",
                "C4_Vertebra_ICRP",
                "SKULL_ICRP",
                "RIB_BONE_ICRP",
            ]

        self.geomet.DSD = 1500
        # I think I can get away with this
        self.geomet.sDetector = self.geomet.dDetector * self.geomet.nDetector

        if head:
            self.geomet.sVoxel = np.array(
                (
                    self.phantom.shape[0] * 3.125,
                    self.phantom.shape[1] / 2,
                    self.phantom.shape[2] / 2,
                )
            )
        else:
            self.geomet.sVoxel = np.array(
                (
                    self.phantom.shape[0] * 3.125,
                    self.phantom.shape[1],
                    self.phantom.shape[2],
                )
            )
        self.geomet.dVoxel = self.geomet.sVoxel / self.geomet.nVoxel

    def analyse_515(self, slc, place, fmt="-"):

        pass

    def reconstruct(self, algo, filt="hamming"):
        '''
        algo, FDK, CGLS ect.

        filt is one of 'hamming','ram_lak','cosine'

        '''

        if algo == "FDK":
            try:
                self.img = tigre.algorithms.FDK(
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
