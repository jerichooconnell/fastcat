import unittest
import os

import fastcat as fc

import numpy as np


class FastcatTest(unittest.TestCase):
    def test_dose_al(self):
        """
        Test if the dose matches MC estimates
        Testing the C 2.5 and W 2.5 MV
        """
        # Intialize
        angles = np.linspace(0, 2 * np.pi, 2)
        s_fc = fc.Spectrum()
        phantom = fc.Catphan_515()
        # Test the C_25
        s_fc.load("C_spectrum_25")
        kernel = fc.Detector(s_fc, "CuGOS-784-micrometer")
        dose = phantom.return_projs(
            kernel,
            s_fc,
            angles,
            nphoton=2e7,
            mgy=0,
            return_dose=True,
            verbose=0,
        )

        #         print(dose[-1])
        # C_25 test
        self.assertAlmostEqual(dose[-1], 0.04099205, places=2)

        # Test the W_25
        s_fc.load("W_spectrum_6")
        kernel = fc.Detector(s_fc, "CuGOS-784-micrometer")
        dose = phantom.return_projs(
            kernel,
            s_fc,
            angles,
            nphoton=2e7,
            mgy=0,
            return_dose=True,
            verbose=0,
        )
        # C_25 test
        self.assertAlmostEqual(dose[-1], 0.1935003, places=2)

    def test_counts_comparison(self):

        """
        Compares the counts in a profile for the 515
        phantom between MC and Fastcat
        3 Tests for the Al W and a kV spectrum
        """

        s2 = fc.calculate_spectrum(120, 12, 3, 50, monitor=None)
        s2.attenuate(0.1, fc.get_mu(z=13))

        spectra = ["Al_spectrum_6", "W_spectrum_6", "kV"]

        data_files = [
            "MC_MV_6Al_counts",
            "MC_MV_6xW_counts",
            "MC_kV_120_counts",
        ]

        angles = np.linspace(0, np.pi * 2, 2)
        phantom = fc.Catphan_515()
        s = fc.Spectrum()

        for ii, spectrum in enumerate(spectra):

            if spectrum == "kV":
                s = s2
            else:
                s.load(spectrum)

            kernel = fc.Detector(s, "CuGOS-784-micrometer")
            counts = np.mean(
                phantom.return_projs(kernel, s, angles, det_on=False, mgy=0.0)[
                    0
                ],
                0,
            )
            MC_counts = np.load(
                os.path.join("test_data", data_files[ii] + ".npy")
            )
            self.assertLess(np.abs(np.max(counts - MC_counts)), 10)

    def test_MTF(self):

        """
        See that the MTF compares to Shi et al. and Star Lack et al.
        Within four percent of shi and the star lack is far out now.
        """
        s = fc.Spectrum()
        s.load("W_spectrum_6")  # Varian_truebeam')
        kernel = fc.Detector(s, "CuGOS-336-micrometer")
        kernel.get_plot_mtf_real(None)

        shi_x = np.load(os.path.join("test_data", "shi_frequencies.npy"))
        shi_y = np.load(os.path.join("test_data", "shi_MTF.npy"))

        int_mtf = np.interp(shi_x, kernel.freq, kernel.mtf)

        #         print(np.max(np.abs(int_mtf - shi_y)))
        # Within 10 percent
        self.assertLess(np.max(np.abs(int_mtf - shi_y / shi_y[0])), 0.04)

        # Star lack
        star_x = np.load(os.path.join("test_data", "star_frequencies.npy"))
        star_y = np.load(os.path.join("test_data", "star_MTF.npy"))

        kernel2 = fc.Detector(s, "CWO-784-micrometer")
        kernel2.get_plot_mtf_real(None)

        int_mtf2 = np.interp(star_x, kernel2.freq, kernel2.mtf)
        #         print(np.max(np.abs(int_mtf2 - star_y/star_y[0])))
        # Within 10 percent
        self.assertLess(np.max(np.abs(int_mtf2 - star_y / star_y[0])), 0.20)

    def test_no_residual(self):
        """
        Making sure that the scan profile is flat without an object
        And that the phantom reproduces the attenuation for water
        """
        spectra = ["Varian_truebeam_phasespace"]
        MV_detectors = ["CuGOS-784-micrometer"]
        angles = np.linspace(np.pi / 2, np.pi * 2, 2)
        phantom = fc.Catphan_404()
        phantom.phan_map = ["1"] * 20
        s = fc.Spectrum()

        s.load(spectra[0])
        s.x = np.array([290, 300, 310, 3000])
        s.y = np.array([0, 1, 0, 0])
        kernel = fc.Detector(s, MV_detectors[0])
        kernel.deposition_interpolated = np.array([1, 0, 0])
        phantom.return_projs(
            kernel, s, angles, mgy=0.0, scat_on=False, convolve_on=False
        )

        # Check that the profile is flat without object
        self.assertLess(np.mean(phantom.proj[0]), 0.01)

        # Check that the correct attenuation is returned
        phantom.phan_map = ["water"] * 20

        phantom.return_projs(
            kernel, s, angles, mgy=0.0, scat_on=False, convolve_on=False
        )
        self.assertAlmostEqual(
            np.max(phantom.proj[0]), 0.97 * 0.1186 * 20 * 10, delta=0.1
        )

    def test_kV_experimental_profile(self):
        """
        Test that the results match an experimental
        profile from Varian Truebeam within
        0.1, this is some error from geometrical distortion in the exp image
        """
        s = fc.calculate_spectrum(100, 14, 5, 200, monitor=None)
        MV_detectors = ["CsI-784-micrometer"]
        angles = np.linspace(np.pi / 2, np.pi * 2, 2)
        phantom = fc.Catphan_404()
        phantom.geomet.DSD = 1530
        kernel = fc.Detector(s, MV_detectors[0])
        kernel.add_focal_spot(1.2)
        phantom.phan_map = [
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
            "CATPHAN_Teflon_revised",
            "air",
            "air",
            "air",
            "air",
        ]
        phantom.return_projs(
            kernel, s, angles, mgy=18, bowtie=True, filter="bowtie_asym2"
        )

        exp_profile = np.load(
            os.path.join("test_data", "kV_experimental_profile.npy")
        )

        self.assertLess(
            np.mean(np.abs(np.roll(phantom.proj[0, 5], -3) - exp_profile)), 1
        )

    def test_MV_experimental_profile(self):
        """
        Test that the results match an experimental
        profile from Varian Truebeam within
        0.1, This one for the MV image.
        """
        spectra = ["Varian_truebeam"]
        MV_detectors = ["CuGOS-784-micrometer"]
        angles = np.linspace(np.pi / 2, np.pi * 2, 2)
        phantom = fc.Catphan_404()
        phantom.geomet.DSD = 1510
        phantom.phan_map = [
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
            "CATPHAN_Teflon_revised",
            "air",
            "air",
            "air",
            "air",
        ]
        s = fc.Spectrum()
        s.load(spectra[0])
        s.x[0] = 1
        s.x[1] = 2
        s.attenuate(0.0, fc.get_mu(z=13))  # 3.7
        kernel = fc.Detector(s, MV_detectors[0])
        kernel.add_focal_spot(1.4)
        phantom.return_projs(
            kernel, s, angles, mgy=0.0, bowtie=True, filter="FF_t0"
        )
        exp_profile = np.load(
            os.path.join("test_data", "MV_experimental_profile.npy")
        )
        self.assertLess(
            np.mean(np.abs(np.roll(phantom.proj[0, 5], -3) - exp_profile)), 1
        )


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
