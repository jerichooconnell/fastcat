#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""xpecgen.py: A module to calculate x-ray spectra generated in tungsten anodes."""

from __future__ import print_function

import math
from bisect import bisect_left
import os
from glob import glob
import warnings
import csv

import numpy as np
from scipy import interpolate, integrate, optimize
import xlsxwriter
from matplotlib import cm
from matplotlib.colors import LogNorm

import tigre
import tigre.algorithms as algs
from scipy.signal import fftconvolve
from scipy.optimize import minimize
from numpy import cos, sin
from builtins import map

try:
    import matplotlib.pyplot as plt

    plt.ion()
    plot_available = True
except ImportError:
    warnings.warn("Unable to import matplotlib. Plotting will be disabled.")
    plot_available = False

__author__ = 'Dih5'
__version__ = "1.3.0"

data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


# --------------------General purpose functions-------------------------#
def return_projs(phantom,kernel,energies,fluence,angles,geo,
                deposition_efficiency_file = 'analysis/2020-06-04-h08m58/EnergyDeposition.npy',
                phantom_mapping = ['air','water','pmma','bone','brain','lung','muscle','pmma'],scaling = None,dose = None):
    '''
    The main function for returning the projections
    '''

    # Don't want to look for zeros
    useful_phantom = phantom != 0

    # These are what I used in the Monte Carlo
    original_energies_keV = np.array([30, 40, 50 ,60, 70, 80 ,90 ,100 ,300 ,500 ,700, 900, 1000 ,2000 ,4000 ,6000])

    mu_en_water = np.array([0.1557, 0.06947,0.04223,0.03190,0.03800,0.02597,0.02554, 0.02546,0.03192,0.03299,0.03244,0.03150,0.03103,0.02608,0.02066,0.01806])

    # Loading the file from the monte carlo
    deposition_summed = np.load(deposition_efficiency_file,allow_pickle=True)

    # This is a scaling factor that I found to work to convert energy deposition to photon probability eta
    deposition_summed = deposition_summed[0]/(original_energies_keV*355)

    # The index of the different materials
    masks = np.zeros([len(phantom_mapping)-1,useful_phantom.shape[0],useful_phantom.shape[1],useful_phantom.shape[2]])
    mapping_functions = []

    # Get the mapping functions for the different tissues to reconstruct the phantom by energy
    for ii in range(1,len(phantom_mapping)):
        
        mapping_functions.append(get_mu(phantom_mapping[ii]))
        masks[ii-1] = phantom == ii

    phantom2 = phantom.copy().astype(np.float32) # Tigre only works with float32

    proj = []
    doses = []

    for energy in original_energies_keV:
        for ii in range(0,len(phantom_mapping)-1):

            phantom2[masks[ii].astype(bool)] = mapping_functions[ii](energy)

        proj.append(np.squeeze(tigre.Ax(phantom2,geo,angles)))
        # Calculate a dose contribution by dividing by 10 since tigre has projections that are a little odd
        doses.append((energy)*(1-np.exp(-(proj[-1][0])/10)))

    # Binning to get the fluence per energy
    large_energies = np.linspace(0,6000,3001)/1000
    fluence_large = np.interp(large_energies,np.array(energies), fluence)

    fluence_small = np.zeros(len(original_energies_keV))

    # Still binning
    for ii, val in enumerate(large_energies):
        
        index = np.argmin(np.abs(original_energies_keV-val*1000))
        fluence_small[index] += fluence_large[ii] 
        
    # Normalize
    fluence_small /= np.sum(fluence_small)
    # Save the scale for later
    deposition_scale = np.sum(deposition_summed) # Thought about using trapz but don't think I need to

    weights_small = fluence_small*deposition_summed

    fluence_original = np.interp(original_energies_keV,energies,fluence)
    fluence_original /= np.sum(fluence_original)

    # Scale by the amount of photons hitting the detector
    deposition_scale = np.trapz(fluence_original*deposition_summed,original_energies_keV)

    # Sum over the image dimesions to get the energy intensity and multiply by fluence
    dose_divided_by_initial_intensity = np.sum(np.sum(doses,1),1)@(fluence_small*mu_en_water)

    # Mass of the phantom there is a times 4 since the detector is 1/4 the size 1000 for mg
    dose_in_mgrays = dose_divided_by_initial_intensity*1.6021766e-13/2.0106*4*1000 # J/MeV 1/kG mGy/Gy

    if dose != 0:
        # Figure out the factor for the SNR
        scaling = (dose)/dose_in_mgrays
        print('Scaled by ', scaling)

    # Need to make sure that the attenuations aren't janky for recon
    weights_small /= np.sum(weights_small)

    # load the arrays
    primary_large = np.load('/home/xcite/xpecgen/tests/primary_kernel_int_larger.npy')
    noise = np.load('/home/xcite/xpecgen/tests/noise_projections.npy')
    primary_512 = (primary_large[:,::2,:,:] + primary_large[:,1::2,:,:])/2
    primary = np.tile(primary_512,[1,1,2,1])

    # sum the noise kernel with the weights
    noise_summed = noise.T @ weights_small
    # weight the primary
    primary_summed = primary.T @ weights_small
    # analytical
    analytical_summed = np.array(proj).T @ weights_small

    # Get a 2d sum
    primary_projections = np.sum(primary_summed,0)
    raw_noise = np.sum(noise_summed,0)
    # analytical_summed = np.mean(analytical_summed,0) # ??

    # Average the primary in 1d
    raw_prime = np.mean(primary_projections,0)
    mean_analytical = np.mean(analytical_summed[:,:,0],1)
    raw_noise = np.mean(raw_noise,0)

    def get_rmse(x):
        return np.sum(np.abs(np.exp(-mean_analytical/x[0])*x[1] - raw_prime))

    def get_rmse_noise(x):
        return np.sum(np.abs(np.exp(-mean_analytical/x[0])*x[1] - raw_noise))

    scale = minimize(get_rmse,[10,32])
    scale_noise = minimize(get_rmse_noise,[10,32])

    raw_proj = np.exp(-mean_analytical/scale.x[0])*scale.x[1]
    # get the noise
    raw_proj = np.exp(-mean_analytical/scale.x[0])*scale.x[1]
    raw_proj_noise = np.exp(-mean_analytical/scale_noise.x[0])*scale_noise.x[1]
    scatter = raw_proj_noise - raw_proj

    # Reshape the projections
    weighted_projs = np.array(proj).transpose([3,2,1,0])

    # Downsize the kernel if it is CWO, bit of a quirk
    if kernel.kernel.shape[0] == 50:
        small_kernel = (kernel.kernel[::2,::2] 
                        + kernel.kernel[1::2,::2] 
                        + kernel.kernel[::2,1::2] 
                        + kernel.kernel[1::2,1::2])/4
    else:
        small_kernel = kernel.kernel
    
    # Normalize the kernel
    kernel_norm = small_kernel/np.sum(small_kernel)
    # log(i/i_0) to get back to intensity
    raw = np.exp(-weighted_projs/scale.x[0])*scale.x[1]
    # Weight the intensity by fluence
    raw_weighted = raw@weights_small
    # Add the already weighted noise
    raw_weighted += np.tile(scatter,[len(angles),raw_weighted.shape[1],1]).T
    # add the poisson noise
    if scaling is not None:
        scaling *= deposition_scale # I think that this is correct
        scaling = 1./scaling
        raw_weighted = np.random.poisson(lam=raw_weighted/scaling)*scaling
    
    filtered = raw_weighted.copy()

    for ii in range(len(angles)):

        filtered[:,:,ii] = fftconvolve(raw_weighted[:,:,ii],kernel_norm, mode = 'same')

    return (-np.log(filtered/scale_noise.x[1])*scale_noise.x[0]).transpose([2,0,1]), dose_in_mgrays

def log_interp_1d(xx, yy, kind='linear'):
    """
    Perform interpolation in log-log scale.

    Args:
        xx (List[float]): x-coordinates of the points.
        yy (List[float]): y-coordinates of the points.
        kind (str or int, optional): The kind of interpolation in the log-log domain. This is passed to
                                     scipy.interpolate.interp1d.

    Returns:
        A function whose call method uses interpolation in log-log scale to find the value at a given point.
    """
    log_x = np.log(xx)
    log_y = np.log(yy)
    # No big difference in efficiency was found when replacing interp1d by
    # UnivariateSpline
    lin_interp = interpolate.interp1d(log_x, log_y, kind=kind)
    return lambda zz: np.exp(lin_interp(np.log(zz)))


# This custom implementation of dblquad is based in the one in numpy
# (Cf. https://github.com/scipy/scipy/blob/v0.16.1/scipy/integrate/quadpack.py#L449 )
# It was modified to work only in rectangular regions (no g(x) nor h(x))
# to set the inner integral epsrel
# and to increase the limit of points taken
def _infunc(x, func, c, d, more_args, epsrel):
    myargs = (x,) + more_args
    return integrate.quad(func, c, d, args=myargs, epsrel=epsrel, limit=2000)[0]


def custom_dblquad(func, a, b, c, d, args=(), epsabs=1.49e-8, epsrel=1.49e-8, maxp1=50, limit=2000):
    """
    A wrapper around numpy's dblquad to restrict it to a rectangular region and to pass arguments to the 'inner'
    integral.

    Args:
        func: The integrand function f(y,x).
        a (float): The lower bound of the second argument in the integrand function.
        b (float): The upper bound of the second argument in the integrand function.
        c (float): The lower bound of the first argument in the integrand function.
        d (float): The upper bound of the first argument in the integrand function.
        args (sequence, optional): extra arguments to pass to func.
        epsabs (float, optional): Absolute tolerance passed directly to the inner 1-D quadrature integration.
        epsrel (float, optional): Relative tolerance of the inner 1-D integrals. Default is 1.49e-8.
        maxp1 (float or int, optional): An upper bound on the number of Chebyshev moments.
        limit (int, optional): Upper bound on the number of cycles (>=3) for use with a sinusoidal weighting and an
                               infinite end-point.

    Returns:
        (tuple): tuple containing:

            y (float): The resultant integral.

            abserr (float): An estimate of the error.

    """
    return integrate.quad(_infunc, a, b, (func, c, d, args, epsrel),
                          epsabs=epsabs, epsrel=epsrel, maxp1=maxp1, limit=limit)


def triangle(x, loc=0, size=0.5, area=1):
    """
    The triangle window function centered in loc, of given size and area, evaluated at a point.

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
    return 0 if abs((x - loc) / size) > 1 else (1 - abs((x - loc) / size)) * abs(area / size)


# --------------------Spectrum model functionality----------------------#
class IndexTracker(object):
    def __init__(self, ax, X):
        self.ax = ax
        ax.set_title('Geometry Viewer')

        self.X = X
        rows, cols, self.slices = X.shape
        self.ind = self.slices//2

        self.im = ax.imshow(self.X[:, :, self.ind])
        self.update()

    def onscroll(self, event):
        #print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        self.im.set_data(self.X[:, :, self.ind])
        self.ax.set_ylabel('slice %s' % self.ind)
        self.im.axes.figure.canvas.draw()

class IndexTracker2(object):
    def __init__(self, ax, X):
        self.ax = ax
        ax.set_title('Projection Viewer')

        self.X = X
        rows, cols, self.slices = X.shape
        self.ind = self.slices//2

        self.im = ax.imshow(self.X[:, :, self.ind])
        self.update()

    def onscroll(self, event):
        #print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        self.im.set_data(self.X[:, :, self.ind])
        self.ax.set_ylabel('Projection %s' % self.ind)
        self.im.axes.figure.canvas.draw()

class Kernel:

    def __init__(self):
        """
        Create an empty spectrum.
        """
        self.kernel = []

    def get_plot(self, place, show_mesh=True, prepare_format=True):
        """
        Prepare a plot of the data in the given place

        Args:
            place: The class whose method plot is called to produce the plot (e.g., matplotlib.pyplot).
            show_mesh (bool): Whether to plot the points over the continuous line as circles.
            prepare_format (bool): Whether to include ticks and labels in the plot.
            peak_shape: The window function used to plot the peaks. See :obj:`triangle` for an example.

        """
        if prepare_format:
            place.tick_params(axis='both', which='major', labelsize=10)
            place.tick_params(axis='both', which='minor', labelsize=8)

        place.imshow(self.kernel, cmap=cm.jet, norm=LogNorm())
        # place.colorbar()
        place.set_title('Point Source Kernel')
        place.set_xlabel('X [pix]')
        place.set_ylabel('Y [pix]')

    def get_plot_mtf(self, place, show_mesh=True, prepare_format=True):
        """
        Prepare a plot of the data in the given place

        Args:
            place: The class whose method plot is called to produce the plot (e.g., matplotlib.pyplot).
            show_mesh (bool): Whether to plot the points over the continuous line as circles.
            prepare_format (bool): Whether to include ticks and labels in the plot.
            peak_shape: The window function used to plot the peaks. See :obj:`triangle` for an example.

        """
        if prepare_format:
            place.tick_params(axis='both', which='major', labelsize=10)
            place.tick_params(axis='both', which='minor', labelsize=8)

        center = int(self.kernel.shape[0]/2)
        lsf = self.kernel[center,:]

        N = self.kernel.shape[0]

        if N < 40:
            T = 0.784
        else:
            T = 0.392
        
        xf = np.linspace(0.0, 1.0/(2.0*T), int(N/2))
        
        mtf = np.absolute(np.fft.fft(lsf))
        mtf_final = np.fft.fftshift(mtf)

        place.plot(xf[:center],mtf_final[center:center + center]/mtf_final[center])
        place.set_title('MTF from Kernel')
        place.set_xlabel('Spacial Frequency [1/mm]')
        place.set_ylabel('MTF')

class Catphan_515:

    def __init__(self,file_path):
        self.phantom = np.load(file_path)
        self.geomet = tigre.geometry_default(high_quality=False,nVoxel=self.phantom.shape)
        self.geomet.nDetector = np.array([64,512])
        self.geomet.dDetector = np.array([0.672, 0.672])

        # I think I can get away with this
        self.geomet.sDetector = self.geomet.dDetector * self.geomet.nDetector    

        self.geomet.sVoxel = np.array((160, 160, 160)) 
        self.geomet.dVoxel = self.geomet.sVoxel/self.geomet.nVoxel 

    def analyse(self,recon_slice):

        def create_mask():

            im = np.zeros([256,256])
            ii = 1

            # CTMAT(x) formel=H2O dichte=x
            LEN = 100

            A0  = 87.7082*np.pi/180
            A1 = 108.3346*np.pi/180
            A2 = 126.6693*np.pi/180
            A3 = 142.7121*np.pi/180
            A4 = 156.4631*np.pi/180
            A5 = 167.9223*np.pi/180
            A6 = 177.0896*np.pi/180
            A7 = 183.9651*np.pi/180
            A8 = 188.5487*np.pi/180

            B0 = 110.6265*np.pi/180
            B1 = 142.7121*np.pi/180
            B2 = 165.6304*np.pi/180
            B3 = 179.3814*np.pi/180

            # Phantom 
            # ++++ module body ++++++++++++++++++++++++++++++++++++++++++++++++++ */                        
            create_circular_mask(x= 0.000,  y= 0.000,  r=1.0, index = ii, image = im)

            ii += 1

            # ++++ supra-slice 1.0% targets +++++++++++++++++++++++++++++++++++++++ */
            create_circular_mask(x= 5*cos(A0),  y= 5*sin(A0),  r=0.75, index = ii, image = im); ii += 1
            create_circular_mask(x= 5*cos(A1),  y= 5*sin(A1),  r=0.45, index = ii, image = im); ii += 1
            create_circular_mask(x= 5*cos(A2),  y= 5*sin(A2),  r=0.40, index = ii, image = im); ii += 1
            create_circular_mask(x= 5*cos(A3),  y= 5*sin(A3),  r=0.35, index = ii, image = im); ii += 1
            create_circular_mask(x= 5*cos(A4),  y= 5*sin(A4),  r=0.30, index = ii, image = im); ii += 1
            create_circular_mask(x= 5*cos(A5),  y= 5*sin(A5),  r=0.25, index = ii, image = im); ii += 1
            create_circular_mask(x= 5*cos(A6),  y= 5*sin(A6),  r=0.20, index = ii, image = im); ii += 1
            create_circular_mask(x= 5*cos(A7),  y= 5*sin(A7),  r=0.15, index = ii, image = im); ii += 1
            create_circular_mask(x= 5*cos(A8),  y= 5*sin(A8),  r=0.10, index = ii, image = im); ii += 1

            return im

        def create_mask_multi(shape):

            im = np.zeros(shape)
            ii = 1

            # CTMAT(x) formel=H2O dichte=x
            LEN = 100

            A0  = 87.7082*np.pi/180
            A1 = 108.3346*np.pi/180
            A2 = 126.6693*np.pi/180
            A3 = 142.7121*np.pi/180
            A4 = 156.4631*np.pi/180
            A5 = 167.9223*np.pi/180
            A6 = 177.0896*np.pi/180
            A7 = 183.9651*np.pi/180
            A8 = 188.5487*np.pi/180

            B0 = 110.6265*np.pi/180
            B1 = 142.7121*np.pi/180
            B2 = 165.6304*np.pi/180
            B3 = 179.3814*np.pi/180

            # Phantom 
            # ++++ module body ++++++++++++++++++++++++++++++++++++++++++++++++++ */                        
            create_circular_mask(x= 0.000,  y= 0.000,  r=1.0, index = ii, image = im)

            ii += 1

            tad = 0.2
            # ++++ supra-slice 1.0% targets +++++++++++++++++++++++++++++++++++++++ */

            create_circular_mask(x= 5*cos(A0),  y= 5*sin(A0),  r=0.75 - tad, index = ii, image = im); ii += 1
            create_circular_mask(x= 5*cos(A0+2/3*np.pi),  y= 5*sin(A0+2/3*np.pi),  r=0.75 - tad, index = ii, image = im); ii += 1
            create_circular_mask(x= 5*cos(A0+4/3*np.pi),  y= 5*sin(A0+4/3*np.pi),  r=0.75 - tad, index = ii, image = im); ii += 1

            create_circular_mask(x= 2.5*cos(B0),  y= 2.5*sin(B0),  r=0.45 - tad, index = ii, image = im); ii += 1
            create_circular_mask(x= 2.5*cos(B0+2/3*np.pi) ,y= 2.5*sin(B0+2/3*np.pi),  r=0.45 - tad  , index = ii, image = im); ii += 1
            create_circular_mask(x= 2.5*cos(B0+4/3*np.pi) ,y= 2.5*sin(B0+4/3*np.pi),  r=0.45 - tad  , index = ii, image = im); ii += 1       
            return im

        def create_circular_mask(x, y, r, index, image):
        
            h,w = image.shape
            
            center = [x*int(w/2)/10 + int(w/2),y*int(h/2)/10 + int(h/2)]

            if center is None: # use the middle of the image
                center = (int(w/2), int(h/2))
            if r is None: # use the smallest distance between the center and image walls
                radius = min(center[0], center[1], w-center[0], h-center[1])

            Y, X = np.ogrid[:h, :w]
            dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

            mask = dist_from_center <= r*int(w/2)/10
            
            
            image[mask] = index

        im = create_mask_multi(recon_slice.shape)

        contrast = []
        noise = []
        cnr = []

        ii = 1

        ref_mean = np.mean(recon_slice[im == ii])
        ref_std = np.std(recon_slice[im == ii])

        for ii in range(2,8):
            
            contrast.append(np.abs(np.mean(recon_slice[im == ii])- ref_mean))
            noise.append(np.std(recon_slice[im == ii]))
            
            cnr.append(contrast[-1]/(np.sqrt(noise[-1]**2)))
            
        rs = np.linspace(0.1,0.45,8)

        return rs, [(contrast[ii]/ref_mean)*100 for ii in range(len(contrast))], cnr

    def analyse_515(self,recon_slice):

        def create_mask(shape):

            im = np.zeros(shape)
            ii = 1

            # CTMAT(x) formel=H2O dichte=x
            LEN = 100

            A0  = 87.7082*np.pi/180
            A1 = 108.3346*np.pi/180
            A2 = 126.6693*np.pi/180
            A3 = 142.7121*np.pi/180
            A4 = 156.4631*np.pi/180
            A5 = 167.9223*np.pi/180
            A6 = 177.0896*np.pi/180
            A7 = 183.9651*np.pi/180
            A8 = 188.5487*np.pi/180

            B0 = 110.6265*np.pi/180
            B1 = 142.7121*np.pi/180
            B2 = 165.6304*np.pi/180
            B3 = 179.3814*np.pi/180

            tad = 0.03

            # Phantom 
            # ++++ module body ++++++++++++++++++++++++++++++++++++++++++++++++++ */                        
            create_circular_mask(x= 0.000,  y= 0.000,  r=-tad + 1.0, index = ii, image = im)

            ii += 1

            # ++++ supra-slice 1.0% targets +++++++++++++++++++++++++++++++++++++++ */
            create_circular_mask(x= 5*cos(A0),  y= 5*sin(A0),  r=-tad + 0.75, index = ii, image = im); ii += 1
            create_circular_mask(x= 5*cos(A1),  y= 5*sin(A1),  r=-tad + 0.45, index = ii, image = im); ii += 1
            create_circular_mask(x= 5*cos(A2),  y= 5*sin(A2),  r=-tad + 0.40, index = ii, image = im); ii += 1
            create_circular_mask(x= 5*cos(A3),  y= 5*sin(A3),  r=-tad + 0.35, index = ii, image = im); ii += 1
            create_circular_mask(x= 5*cos(A4),  y= 5*sin(A4),  r=-tad + 0.30, index = ii, image = im); ii += 1
            create_circular_mask(x= 5*cos(A5),  y= 5*sin(A5),  r=-tad + 0.25, index = ii, image = im); ii += 1
            create_circular_mask(x= 5*cos(A6),  y= 5*sin(A6),  r=-tad + 0.20, index = ii, image = im); ii += 1
            create_circular_mask(x= 5*cos(A7),  y= 5*sin(A7),  r=-tad + 0.15, index = ii, image = im); ii += 1
            create_circular_mask(x= 5*cos(A8),  y= 5*sin(A8),  r=-tad + 0.10, index = ii, image = im); ii += 1


            # ++++ supra-slice 0.3% targets +++++++++++++++++++++++++++++++++++++++ */
            create_circular_mask(x= 5*cos(A0+2/3*np.pi),  y= 5*sin(A0+2/3*np.pi),  r=-tad + 0.75, index = ii, image = im); ii += 1
            create_circular_mask(x= 5*cos(A1+2/3*np.pi),  y= 5*sin(A1+2/3*np.pi),  r=-tad + 0.45, index = ii, image = im); ii += 1
            create_circular_mask(x= 5*cos(A2+2/3*np.pi),  y= 5*sin(A2+2/3*np.pi),  r=-tad + 0.40, index = ii, image = im); ii += 1
            create_circular_mask(x= 5*cos(A3+2/3*np.pi),  y= 5*sin(A3+2/3*np.pi),  r=-tad + 0.35, index = ii, image = im); ii += 1
            create_circular_mask(x= 5*cos(A4+2/3*np.pi),  y= 5*sin(A4+2/3*np.pi),  r=-tad + 0.30, index = ii, image = im); ii += 1
            create_circular_mask(x= 5*cos(A5+2/3*np.pi),  y= 5*sin(A5+2/3*np.pi),  r=-tad + 0.25, index = ii, image = im); ii += 1
            create_circular_mask(x= 5*cos(A6+2/3*np.pi),  y= 5*sin(A6+2/3*np.pi),  r=-tad + 0.20, index = ii, image = im); ii += 1
            create_circular_mask(x= 5*cos(A7+2/3*np.pi),  y= 5*sin(A7+2/3*np.pi),  r=-tad + 0.15, index = ii, image = im); ii += 1
            create_circular_mask(x= 5*cos(A8+2/3*np.pi),  y= 5*sin(A8+2/3*np.pi),  r=-tad + 0.10, index = ii, image = im); ii += 1


            # ++++ supra-slice 0.5% targets +++++++++++++++++++++++++++++++++++++++ */
            create_circular_mask(x= 5*cos(A0+4/3*np.pi),  y= 5*sin(A0+4/3*np.pi),  r=-tad + 0.75, index = ii, image = im); ii += 1
            create_circular_mask(x= 5*cos(A1+4/3*np.pi),  y= 5*sin(A1+4/3*np.pi),  r=-tad + 0.45, index = ii, image = im); ii += 1
            create_circular_mask(x= 5*cos(A2+4/3*np.pi),  y= 5*sin(A2+4/3*np.pi),  r=-tad + 0.40, index = ii, image = im); ii += 1
            create_circular_mask(x= 5*cos(A3+4/3*np.pi),  y= 5*sin(A3+4/3*np.pi),  r=-tad + 0.35, index = ii, image = im); ii += 1
            create_circular_mask(x= 5*cos(A4+4/3*np.pi),  y= 5*sin(A4+4/3*np.pi),  r=-tad + 0.30, index = ii, image = im); ii += 1
            create_circular_mask(x= 5*cos(A5+4/3*np.pi),  y= 5*sin(A5+4/3*np.pi),  r=-tad + 0.25, index = ii, image = im); ii += 1
            create_circular_mask(x= 5*cos(A6+4/3*np.pi),  y= 5*sin(A6+4/3*np.pi),  r=-tad + 0.20, index = ii, image = im); ii += 1
            create_circular_mask(x= 5*cos(A7+4/3*np.pi),  y= 5*sin(A7+4/3*np.pi),  r=-tad + 0.15, index = ii, image = im); ii += 1
            create_circular_mask(x= 5*cos(A8+4/3*np.pi),  y= 5*sin(A8+4/3*np.pi),  r=-tad + 0.10, index = ii, image = im); ii += 1

            # ++++ subslice 1.0% targets 7mm long +++++++++++++++++++++++++++++++++ */
            create_circular_mask(x= 2.5*cos(B0),  y= 2.5*sin(B0),  r=-tad + 0.45, index = ii, image = im); ii += 1
            create_circular_mask(x= 2.5*cos(B1),  y= 2.5*sin(B1),  r=-tad + 0.35, index = ii, image = im); ii += 1
            create_circular_mask(x= 2.5*cos(B2),  y= 2.5*sin(B2),  r=-tad + 0.25, index = ii, image = im); ii += 1
            create_circular_mask(x= 2.5*cos(B3),  y= 2.5*sin(B3),  r=-tad + 0.15, index = ii, image = im); ii += 1


            # ++++ subslice 1.0% targets 3mm long +++++++++++++++++++++++++++++++++ */
            create_circular_mask(x= 2.5*cos(B0+2/3*np.pi) ,y= 2.5*sin(B0+2/3*np.pi),  r=-tad + 0.45  , index = ii, image = im); ii += 1
            create_circular_mask(x= 2.5*cos(B1+2/3*np.pi) ,y= 2.5*sin(B1+2/3*np.pi),  r=-tad + 0.35  , index = ii, image = im); ii += 1
            create_circular_mask(x= 2.5*cos(B2+2/3*np.pi) ,y= 2.5*sin(B2+2/3*np.pi),  r=-tad + 0.25  , index = ii, image = im); ii += 1
            create_circular_mask(x= 2.5*cos(B3+2/3*np.pi) ,y= 2.5*sin(B3+2/3*np.pi),  r=-tad + 0.15  , index = ii, image = im); ii += 1


            # ++++ subslice 1.0% targets 5mm long +++++++++++++++++++++++++++++++++ */
            create_circular_mask(x= 2.5*cos(B0+4/3*np.pi) ,y= 2.5*sin(B0+4/3*np.pi),  r=-tad + 0.45  , index = ii, image = im); ii += 1
            create_circular_mask(x= 2.5*cos(B1+4/3*np.pi) ,y= 2.5*sin(B1+4/3*np.pi),  r=-tad + 0.35  , index = ii, image = im); ii += 1
            create_circular_mask(x= 2.5*cos(B2+4/3*np.pi) ,y= 2.5*sin(B2+4/3*np.pi),  r=-tad + 0.25  , index = ii, image = im); ii += 1
            create_circular_mask(x= 2.5*cos(B3+4/3*np.pi) ,y= 2.5*sin(B3+4/3*np.pi),  r=-tad + 0.15  , index = ii, image = im); ii += 1

            return im

        def create_mask_multi(shape):

            im = np.zeros(shape)
            ii = 1

            # CTMAT(x) formel=H2O dichte=x
            LEN = 100

            A0  = 87.7082*np.pi/180
            A1 = 108.3346*np.pi/180
            A2 = 126.6693*np.pi/180
            A3 = 142.7121*np.pi/180
            A4 = 156.4631*np.pi/180
            A5 = 167.9223*np.pi/180
            A6 = 177.0896*np.pi/180
            A7 = 183.9651*np.pi/180
            A8 = 188.5487*np.pi/180
            B0 = 110.6265*np.pi/180
            B1 = 142.7121*np.pi/180
            B2 = 165.6304*np.pi/180
            B3 = 179.3814*np.pi/180

            # Phantom 
            # ++++ module body ++++++++++++++++++++++++++++++++++++++++++++++++++ */                        
            create_circular_mask(x= 0.000,  y= 0.000,  r=1.0, index = ii, image = im)

            ii += 1

            tad = 0.2
            # ++++ supra-slice 1.0% targets +++++++++++++++++++++++++++++++++++++++ */

            create_circular_mask(x= 5*cos(A0),  y= 5*sin(A0),  r=0.75 - tad, index = ii, image = im); ii += 1
            create_circular_mask(x= 5*cos(A0+2/3*np.pi),  y= 5*sin(A0+2/3*np.pi),  r=0.75 - tad, index = ii, image = im); ii += 1
            create_circular_mask(x= 5*cos(A0+4/3*np.pi),  y= 5*sin(A0+4/3*np.pi),  r=0.75 - tad, index = ii, image = im); ii += 1

            create_circular_mask(x= 2.5*cos(B0),  y= 2.5*sin(B0),  r=0.45 - tad, index = ii, image = im); ii += 1
            create_circular_mask(x= 2.5*cos(B0+2/3*np.pi) ,y= 2.5*sin(B0+2/3*np.pi),  r=0.45 - tad  , index = ii, image = im); ii += 1
            create_circular_mask(x= 2.5*cos(B0+4/3*np.pi) ,y= 2.5*sin(B0+4/3*np.pi),  r=0.45 - tad  , index = ii, image = im); ii += 1       
            return im

        def create_circular_mask(x, y, r, index, image):
        
            h,w = image.shape
            
            center = [x*int(w/2)/10 + int(w/2),y*int(h/2)/10 + int(h/2)]

            if center is None: # use the middle of the image
                center = (int(w/2), int(h/2))
            if r is None: # use the smallest distance between the center and image walls
                radius = min(center[0], center[1], w-center[0], h-center[1])

            Y, X = np.ogrid[:h, :w]
            dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

            mask = dist_from_center <= r*int(w/2)/10
            
            
            image[mask] = index

        im = create_mask(recon_slice.shape)

        contrast = []
        noise = []
        cnr = []

        ii = 1

        ref_mean = np.mean(recon_slice[im == ii])
        ref_std = np.std(recon_slice[im == ii])

        for ii in range(2,int(np.max(im)+1)):
            
            contrast.append(np.abs(np.mean(recon_slice[im == ii])- ref_mean))
            noise.append(np.std(recon_slice[im == ii]))
            
            cnr.append(contrast[-1]/(np.sqrt(noise[-1]**2)))
            
        rs = np.linspace(0.1,0.45,8)

        return_im = False

        if return_im:
            return rs, [(contrast[ii]/ref_mean)*100 for ii in range(len(contrast))], cnr, im
        else:
            return rs, [(contrast[ii]/ref_mean)*100 for ii in range(len(contrast))], cnr
        

class Spectrum:
    """
    Set of 2D points and discrete components representing a spectrum.

    A Spectrum can be multiplied by a scalar (int, float...) to increase its counts in such a factor.
    Two spectra can be added if they share their x axes and their discrete component positions.

    Note: When two spectrum are added it is not checked it that addition makes sense. It is the user's responsibility to
          do so.

    Attributes:
        x (:obj:`numpy.ndarray`): x coordinates (energy) describing the continuum part of the spectrum.
        y (:obj:`numpy.ndarray`): y coordinates (pdf) describing the continuum part of the spectrum.
        discrete (List[List[float]]): discrete components of the spectrum, each of the form [x, num, rel_x] where:

            * x is the mean position of the peak.
            * num is the number of particles in the peak.
            * rel_x is a characteristic distance where it should extend. The exact meaning depends on the windows function.

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
            An interpolation function representing the continuous part of the spectrum.

        """
        return interpolate.interp1d(self.x, self.y, bounds_error=False, fill_value=0)

    def get_points(self, peak_shape=triangle, num_discrete=10):
        """
        Returns two lists of coordinates x y representing the whole spectrum, both the continuous and discrete components.
        The mesh is chosen by extending x to include details of the discrete peaks.

        Args:
            peak_shape: The window function used to calculate the peaks. See :obj:`triangle` for an example.
            num_discrete: Number of points that are added to mesh in each peak.

        Returns:
            (tuple): tuple containing:

                x2 (List[float]): The list of x coordinates (energy) in the whole spectrum.

                y2 (List[float]): The list of y coordinates (density) in the whole spectrum.

        """
        if peak_shape is None or self.discrete == []:
            return self.x[:], self.y[:]
        # A mesh for each discrete component:
        discrete_mesh = np.concatenate(list(map(lambda x: np.linspace(
            x[0] - x[2], x[0] + x[2], num=num_discrete, endpoint=True), self.discrete)))
        x2 = sorted(np.concatenate((discrete_mesh, self.x)))
        f = self.get_continuous_function()
        peak = np.vectorize(peak_shape)

        def g(x):
            t = 0
            for l in self.discrete:
                t += peak(x, loc=l[0], size=l[2]) * l[1]
            return t

        y2 = [f(x) + g(x) for x in x2]
        return x2, y2

    def get_plot(self, place, show_mesh=True, prepare_format=True, peak_shape=triangle):
        """
        Prepare a plot of the data in the given place

        Args:
            place: The class whose method plot is called to produce the plot (e.g., matplotlib.pyplot).
            show_mesh (bool): Whether to plot the points over the continuous line as circles.
            prepare_format (bool): Whether to include ticks and labels in the plot.
            peak_shape: The window function used to plot the peaks. See :obj:`triangle` for an example.

        """
        if prepare_format:
            place.tick_params(axis='both', which='major', labelsize=10)
            place.tick_params(axis='both', which='minor', labelsize=8)
            place.set_xlabel('E', fontsize=10, fontweight='bold')
            place.set_ylabel('f(E)', fontsize=10, fontweight='bold')

        x2, y2 = self.get_points(peak_shape=peak_shape)
        if show_mesh:
            place.plot(self.x, self.y, 'bo', x2, y2, 'b-')
        else:
            place.plot(x2, y2, 'b-')

    def show_plot(self, show_mesh=True, block=True):
        """
        Prepare the plot of the data and show it in matplotlib window.

        Args:
            show_mesh (bool): Whether to plot the points over the continuous line as circles.
            block (bool): Whether the plot is blocking or non blocking.

        """
        if plot_available:
            plt.clf()
            self.get_plot(plt, show_mesh=show_mesh, prepare_format=False)
            plt.xlabel("E")
            plt.ylabel("f(E)")
            plt.gcf().canvas.set_window_title("".join(('xpecgen v', __version__)))
            plt.show(block=block)
        else:
            warnings.warn("Asked for a plot but matplotlib could not be imported.")

    def export_csv(self, route="a.csv", peak_shape=triangle, transpose=False):
        """
        Export the data to a csv file (comma-separated values).

        Args:
            route (str): The route where the file will be saved.
            peak_shape: The window function used to plot the peaks. See :obj:`triangle` for an example.
            transpose (bool): True to write in two columns, False in two rows.

        """
        x2, y2 = self.get_points(peak_shape=peak_shape)
        with open(route, 'w') as csvfile:
            w = csv.writer(csvfile, dialect='excel')
            if transpose:
                w.writerows([list(a) for a in zip(*[x2, y2])])
            else:
                w.writerow(x2)
                w.writerow(y2)

    def export_xlsx(self, route="a.xlsx", peak_shape=triangle, markers=False):
        """
        Export the data to a xlsx file (Excel format).

        Args:
            route (str): The route where the file will be saved.
            peak_shape: The window function used to plot the peaks. See :obj:`triangle` for an example.
            markers (bool): Whether to use markers or a continuous line in the plot in the file.

        """
        x2, y2 = self.get_points(peak_shape=peak_shape)

        workbook = xlsxwriter.Workbook(route)
        worksheet = workbook.add_worksheet()
        bold = workbook.add_format({'bold': 1})
        worksheet.write(0, 0, "Energy (keV)", bold)
        worksheet.write(0, 1, "Photon density (1/keV)", bold)
        worksheet.write_column('A2', x2)
        worksheet.write_column('B2', y2)

        # Add a plot
        if markers:
            chart = workbook.add_chart(
                {'type': 'scatter', 'subtype': 'straight_with_markers'})
        else:
            chart = workbook.add_chart(
                {'type': 'scatter', 'subtype': 'straight'})

        chart.add_series({
            'name': '=Sheet1!$B$1',
            'categories': '=Sheet1!$A$2:$A$' + str(len(x2) + 1),
            'values': '=Sheet1!$B$2:$B$' + str(len(y2) + 1),
        })
        chart.set_title({'name': 'Emission spectrum'})
        chart.set_x_axis(
            {'name': 'Energy (keV)', 'min': 0, 'max': str(x2[-1])})
        chart.set_y_axis({'name': 'Photon density (1/keV)'})
        chart.set_legend({'position': 'none'})
        chart.set_style(11)

        worksheet.insert_chart('D3', chart, {'x_offset': 25, 'y_offset': 10})

        workbook.close()

    def get_norm(self, weight=None):
        """
        Return the norm of the spectrum using a weighting function.

        Args:
            weight: A function used as a weight to calculate the norm. Typical examples are:

                * weight(E)=1 [Photon number]
                * weight(E)=E [Energy]
                * weight(E)=fluence2Dose(E) [Dose]

        Returns:
            (float): The calculated norm.

        """
        if weight is None:
            w = lambda x: 1
        else:
            w = weight
        y2 = list(map(lambda x, y: w(x) * y, self.x, self.y))
        
        return integrate.simps(y2, x=self.x) + sum([w(a[0]) * a[1] for a in self.discrete])

    def set_norm(self, value=1, weight=None):
        """
        Set the norm of the spectrum using a weighting function.

        Args:
            value (float): The norm of the modified spectrum in the given convention.
            weight: A function used as a weight to calculate the norm. Typical examples are:

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

        This method calculates the depth of a material needed for a certain dosimetric magnitude to decrease in a given factor.

        Args:
            value (float): The factor the desired magnitude is decreased. Must be in [0, 1].
            weight: A function used as a weight to calculate the norm. Typical examples are:

                * weight(E)=1 [Photon number]
                * weight(E)=E [Energy]
                * weight(E)=fluence2Dose(E) [Dose]
            mu: The energy absorption coefficient as a function of energy.
            energy_min (float): A low-energy cutoff to use in the calculation.

        Returns:
            (float): The generalized hvl in cm.

        """
        # TODO: (?) Cut characteristic if below cutoff. However, such a high cutoff
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
            discrete3 = [[a[0], weight(a[0]) * a[1] / n2] for a in self.discrete]
            # Now we only need to add attenuation as a function of depth
            f = lambda t: integrate.simps(list(map(lambda a, b: b * math.exp(-mu(a) * t), x, y3)), x=x) + sum(
                [c[1] * math.exp(-mu(c[0]) * t) for c in discrete3]) - value
            # Search the order of magnitude of the root (using the fact that f is
            # monotonically decreasing)
            a = 1.0
            if f(a) > 0:
                while f(a) > 0:
                    a *= 10.0
                # Now f(a)<=0 and f(a*0.1)>0
                return optimize.brentq(f, a * 0.1, a)
            else:
                while f(a) < 0:
                    a *= 0.1
                # Now f(a)>=0 and f(a*10)<0
                return optimize.brentq(f, a, a * 10.0)

        except ValueError:
            warnings.warn("Interpolation boundary error")
            return 0

    def attenuate(self, depth=1, mu=lambda x: 1):
        """
        Attenuate the spectrum as if it passed thorough a given depth of material with attenuation described by a given
        attenuation coefficient. Consistent units should be used.

        Args:
            depth: The amount of material (typically in cm).
            mu: The energy-dependent absorption coefficient (typically in cm^-1).


        """

        self.y = list(
            map(lambda x, y: y * math.exp(-mu(x) * depth), self.x, self.y))
        self.discrete = list(
            map(lambda l: [l[0], l[1] * math.exp(-mu(l[0]) * depth), l[2]], self.discrete))

    def __add__(self, other):
        """Add two instances, assuming that makes sense."""
        if not isinstance(other, Spectrum):  # so s+0=s and sum([s1, s2,...]) makes sense
            return self
        s = Spectrum()
        s.x = self.x
        s.y = [a + b for a, b in zip(self.y, other.y)]
        s.discrete = [[a[0], a[1] + b[1], a[2]] for a, b in zip(self.discrete, other.discrete)]
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


# --------------------Spectrum calculation functionality----------------#


def get_fluence(e_0=100.0):
    """
    Returns a function representing the electron fluence with the distance in CSDA units.

    Args:
        e_0 (float): The kinetic energy whose CSDA range is used to scale the distances.

    Returns:
        A function representing fluence(x,u) with x in CSDA units.

    """
    # List of available energies
    e0_str_list = list(map(lambda x: (os.path.split(x)[1]).split(".csv")[
        0], glob(os.path.join(data_path, "fluence", "*.csv"))))
    e0_list = sorted(list(map(int, list(filter(str.isdigit, e0_str_list)))))

    e_closest = min(e0_list, key=lambda x: abs(x - e_0))

    with open(os.path.join(data_path, "fluence/grid.csv"), 'r') as csvfile:
        r = csv.reader(csvfile, delimiter=' ', quotechar='|',
                       quoting=csv.QUOTE_MINIMAL)
        t = next(r)
        x = np.array([float(a) for a in t[0].split(",")])
        t = next(r)
        u = np.array([float(a) for a in t[0].split(",")])
    t = []
    with open(os.path.join(data_path, "fluence", "".join([str(e_closest), ".csv"])), 'r') as csvfile:
        r = csv.reader(csvfile, delimiter=' ', quotechar='|',
                       quoting=csv.QUOTE_MINIMAL)
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
        A function representing cross_section(e_g,u) in mb/keV, with e_g in keV.

    """
    # NOTE: Data is given for E0>1keV. CS values below this level should be used with caution.
    # The default behaviour is to keep it constant
    with open(os.path.join(data_path, "cs/grid.csv"), 'r') as csvfile:
        r = csv.reader(csvfile, delimiter=' ', quotechar='|',
                       quoting=csv.QUOTE_MINIMAL)
        t = next(r)
        e_e = np.array([float(a) for a in t[0].split(",")])
        log_e_e = np.log10(e_e)
        t = next(r)
        k = np.array([float(a) for a in t[0].split(",")])
    t = []
    with open(os.path.join(data_path, "cs/%d.csv" % z), 'r') as csvfile:
        r = csv.reader(csvfile, delimiter=' ', quotechar='|',
                       quoting=csv.QUOTE_MINIMAL)
        for row in r:
            t.append([float(a) for a in row[0].split(",")])
    t = np.array(t)
    scaled = interpolate.RectBivariateSpline(log_e_e, k, t, kx=3, ky=1)
    m_electron = 511
    z2 = z * z
    return lambda e_g, u: (u * e_0 + m_electron) ** 2 * z2 / (u * e_0 * e_g * (u * e_0 + 2 * m_electron)) * (
        scaled(np.log10(u * e_0), e_g / (u * e_0)))


def get_mu(z=74):
    """
    Returns a function representing an energy-dependent attenuation coefficient.

    Args:
        z (int or str): The identifier of the material in the data folder, typically the atomic number.

    Returns:
        The attenuation coefficient mu(E) in cm^-1 as a function of the energy measured in keV.

    """
    with open(os.path.join(data_path, "mu", "".join([str(z), ".csv"])), 'r') as csvfile:
        r = csv.reader(csvfile, delimiter=' ', quotechar='|',
                       quoting=csv.QUOTE_MINIMAL)
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
        The CSDA range in cm in tungsten as a function of the electron kinetic energy in keV.

    """
    with open(os.path.join(data_path, "csda/%d.csv" % z), 'r') as csvfile:
        r = csv.reader(csvfile, delimiter=' ', quotechar='|',
                       quoting=csv.QUOTE_MINIMAL)
        t = next(r)
        x = [float(a) for a in t[0].split(",")]
        t = next(r)
        y = [float(a) for a in t[0].split(",")]
    return interpolate.interp1d(x, y, kind='linear')


def get_mu_csda(e_0, z=74):
    """
    Returns a function representing the CSDA-scaled energy-dependent attenuation coefficient in tungsten.

    Args:
        e_0 (float): The electron initial kinetic energy.
        z (int): Atomic number of the material.

    Returns:
        The attenuation coefficient mu(E) in CSDA units as a function of the energy measured in keV.

    """
    mu = get_mu(z)
    csda = get_csda(z=z)(e_0)
    return lambda e: mu(e) * csda


def get_fluence_to_dose():
    """
    Returns a function representing the weighting factor which converts fluence to dose.

    Returns:
        A function representing the weighting factor which converts fluence to dose in Gy * cm^2.

    """
    with open(os.path.join(data_path, "fluence2dose/f2d.csv"), 'r') as csvfile:
        r = csv.reader(csvfile, delimiter=' ', quotechar='|',
                       quoting=csv.QUOTE_MINIMAL)
        t = next(r)
        x = [float(a) for a in t[0].split(",")]
        t = next(r)
        y = [float(a) for a in t[0].split(",")]
    return interpolate.interp1d(x, y, kind='linear')

def update_fluence(mv_file,value):

    # Returns the noise estimate in photons per pixel in ver specific geo

    target_list_load = list(map(lambda x: (os.path.split(x)[1]).split(
    ".txt")[0], glob(os.path.join(data_path, "MV_spectra", "*.txt"))))
    
    # photons per coulomb
    fluences_per_ma = np.array([

        413742,
        2675367,
        2890691,
        317988,
        1476616,
        258593

    ])*15000000/6.242E9*value #

    print(31/50000000 * fluences_per_ma[target_list_load.index(mv_file)])

    return 31/50000000 * fluences_per_ma[target_list_load.index(mv_file)]

def get_source_function(fluence, cs, mu, theta, e_g, phi=0.0):
    """
    Returns the attenuated source function (Eq. 2 in the paper) for the given parameters.

    An E_0-dependent factor (the fraction found there) is excluded. However, the E_0 dependence is removed in
    integrate_source.

    Args:
        fluence: The function representing the fluence.
        cs: The function representing the bremsstrahlung cross-section.
        mu: The function representing the attenuation coefficient.
        theta (float): The emission angle in degrees, the anode's normal being at 90º.
        e_g (float): The emitted photon energy in keV.
        phi (float): The elevation angle in degrees, the anode's normal being at 12º.

    Returns:
        The attenuated source function s(u,x).

    """
    factor = -mu(e_g) / math.sin(math.radians(theta)) / math.cos(math.radians(phi))
    return lambda u, x: fluence(x, u) * cs(e_g, u) * math.exp(factor * x)


def integrate_source(fluence, cs, mu, theta, e_g, e_0, phi=0.0, x_min=0.0, x_max=0.6, epsrel=0.1, z=74):
    """
    Find the integral of the attenuated source function.

    An E_0-independent factor is excluded (i.e., the E_0 dependence on get_source_function is taken into account here).

    Args:
        fluence: The function representing the fluence.
        cs: The function representing the bremsstrahlung cross-section.
        mu: The function representing the attenuation coefficient.
        theta (float): The emission angle in degrees, the anode's normal being at 90º.
        e_g: (float): The emitted photon energy in keV.
        e_0 (float): The electron initial kinetic energy.
        phi (float): The elevation angle in degrees, the anode's normal being at 12º.
        x_min: The lower-bound of the integral in depth, scaled by the CSDA range.
        x_max: The upper-bound of the integral in depth, scaled by the CSDA range.
        epsrel: The relative tolerance of the integral.
        z (int): Atomic number of the material.

    Returns:
        float: The value of the integral.
    """
    if e_g >= e_0:
        return 0
    f = get_source_function(fluence, cs, mu, theta, e_g, phi=phi)
    (y, y_err) = custom_dblquad(f, x_min, x_max, e_g / e_0, 1, epsrel=epsrel, limit=100)
    # The factor includes n_med, its units being 1/(mb * r_CSDA). We only take into account the r_CSDA dependence.
    y *= get_csda(z=z)(e_0)
    return y


def add_char_radiation(s, method="fraction_above_poly"):
    """
    Adds characteristic radiation to a calculated bremsstrahlung spectrum, assuming it is a tungsten-generated spectrum

    If a discrete component already exists in the spectrum, it is replaced.

    Args:
        s (:obj:`Spectrum`): The spectrum whose discrete component is recalculated.
        method (str): The method to use to calculate the discrete component. Available methods include:

            * 'fraction_above_linear': Use a linear relation between bremsstrahlung above the K-edge and peaks.
            * 'fraction_above_poly': Use polynomial fits between bremsstrahlung above the K-edge and peaks.

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
            print(
                "WARNING: Unknown char radiation calculation method. Using fraction_above_poly")
        s.discrete.append([58.65, (0.1912 * fraction_above - 0.00615 *
                                   fraction_above ** 2 - 0.1279 * fraction_above ** 3) * norm, 1])
        s.discrete.append([67.244, (0.04239 * fraction_above + 0.002003 *
                                    fraction_above ** 2 - 0.02356 * fraction_above ** 3) * norm, 1])
        s.discrete.append([69.067, (0.01437 * fraction_above + 0.002346 *
                                    fraction_above ** 2 - 0.009332 * fraction_above ** 3) * norm, 1])

    return


def console_monitor(a, b):
    """
    Simple monitor function which can be used with :obj:`calculate_spectrum`.

    Prints in stdout 'a/b'.

    Args:
        a: An object representing the completed amount (e.g., a number representing a part...).
        b: An object representing the total amount (... of a number representing a total).


    """
    print("Calculation: ", a, "/", b)


def calculate_spectrum_mesh(e_0, theta, mesh, phi=0.0, epsrel=0.2, monitor=console_monitor, z=74):
    """
    Calculates the x-ray spectrum for given parameters.
    Characteristic peaks are also calculated by add_char_radiation, which is called with the default parameters.

    Args:
        e_0 (float): Electron kinetic energy in keV
        theta (float): X-ray emission angle in degrees, the normal being at 90º
        mesh (list of float or ndarray): The photon energies where the integral will be evaluated
        phi (float): X-ray emission elevation angle in degrees.
        epsrel (float): The tolerance parameter used in numeric integration.
        monitor: A function to be called after each iteration with arguments finished_count, total_count. See for example :obj:`console_monitor`.
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
        s.y.append(integrate_source(fluence, cs, mu, theta, e_g, e_0, phi=phi, epsrel=epsrel, z=z))
        if monitor is not None:
            monitor(i + 1, mesh_len)

    if z == 74:
        add_char_radiation(s)

    return s


def calculate_spectrum(e_0, theta, e_min, num_e, phi=0.0, epsrel=0.2, monitor=console_monitor, z=74):
    """
    Calculates the x-ray spectrum for given parameters.
    Characteristic peaks are also calculated by add_char_radiation, which is called with the default parameters.

    Args:
        e_0 (float): Electron kinetic energy in keV
        theta (float): X-ray emission angle in degrees, the normal being at 90º
        e_min (float): Minimum kinetic energy to calculate in the spectrum in keV
        num_e (int): Number of points to calculate in the spectrum
        phi (float): X-ray emission elevation angle in degrees.
        epsrel (float): The tolerance parameter used in numeric integration.
        monitor: A function to be called after each iteration with arguments finished_count, total_count. See for example :obj:`console_monitor`.
        z (int): Atomic number of the material.

    Returns:
        :obj:`Spectrum`: The calculated spectrum

    """
    return calculate_spectrum_mesh(e_0, theta, np.linspace(e_min, e_0, num=num_e, endpoint=True), phi=phi,
                                   epsrel=epsrel, monitor=monitor, z=z)


def cli():
    import argparse
    import sys

    parser = argparse.ArgumentParser(description='Calculate a bremsstrahlung spectrum.')
    parser.add_argument('e_0', metavar='E0', type=float,
                        help='Electron kinetic energy in keV')
    parser.add_argument('theta', metavar='theta', type=float, default=12,
                        help="X-ray emission angle in degrees, the anode's normal being at 90º.")

    parser.add_argument('--phi', metavar='phi', type=float, default=0,
                        help="X-ray emission altitude in degrees, the anode's normal being at 0º.")

    parser.add_argument('--z', metavar='z', type=int, default=74,
                        help="Atomic number of the material (characteristic radiation is only available for z=74).")

    parser.add_argument('--e_min', metavar='e_min', type=float, default=3.0,
                        help="Minimum kinetic energy in keV in the bremsstrahlung calculation.")

    parser.add_argument('--n_points', metavar='n_points', type=int, default=50,
                        help="Number of points used in the bremsstrahlung calculation.")

    parser.add_argument('--mesh', metavar='e_i', type=float, nargs='+',
                        help="Energy mesh where the bremsstrahlung will be calculated. "
                             "Overrides e_min and n_points parameters.")

    parser.add_argument('--epsrel', metavar='tolerance', type=float, default=0.5,
                        help="Numerical tolerance in integration.")

    parser.add_argument('-o', '--output', metavar='path', type=str,
                        help="Output file. Available formats are csv, xlsx, and pkl, selected by the file extension. "
                             "pkl appends objects using the pickle module. Note you have to import the Spectrum class "
                             " INTO THE NAMESPACE (e.g., from xpecgen.xpecgen import Spectrum) to load them. "
                             "If this argument is not provided, points are written to the standard output and "
                             "calculation monitor is not displayed.")

    parser.add_argument('--overwrite', action="store_true",
                        help="If this flag is set and the output is a pkl file, overwrite its content instead of "
                             "appending.")

    args = parser.parse_args()

    if args.output is not None:
        if "." not in args.output:
            print("Output file format unknown", file=sys.stderr)
            exit(-1)
        else:
            ext = args.output.split(".")[-1].lower()
            if ext not in ["csv", "xlsx", "pkl"]:
                print("Output file format unknown", file=sys.stderr)
                exit(-1)
        monitor = console_monitor
    else:
        monitor = None

    if args.mesh is None:
        mesh = np.linspace(args.e_min, args.e_0, num=args.n_points, endpoint=True)
    else:
        mesh = args.mesh

    s = calculate_spectrum_mesh(args.e_0, args.theta, mesh, phi=args.phi, epsrel=args.epsrel, monitor=monitor, z=args.z)
    x2, y2 = s.get_points()

    if args.output is None:
        [print("%.6g, %.6g" % (x, y)) for x, y in zip(x2, y2)]
    elif ext == "csv":
        s.export_csv(args.output)
    elif ext == "xlsx":
        s.export_xlsx(args.output)
    elif ext == "pkl":
        import pickle
        print(args.overwrite)
        if args.overwrite:
            mode = "wb"
        else:
            mode = "ab"

        with open(args.output, mode) as output:
            pickle.dump(s, output, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    cli()

def get_kernel(spectrum,
            dump_files = 'analysis/2020-06-06-h10m44/*.npy',
            deposition_efficiency_file = 'analysis/2020-06-04-h08m58/EnergyDeposition.npy',
            Verbose = False):

    files = sorted(glob(dump_files))

    for ii, file in enumerate(files):
        
        if ii == 0:
            
            # Make the first entry zeros
            first_kernel = np.load(file)
            kernels = np.zeros([len(files)+1,first_kernel.shape[0],first_kernel.shape[1]]) 
            
        kernels[ii+1] = np.load(file)

    fluence = spectrum.y
    energies = spectrum.x
            
    fluence = fluence/sum(fluence)
    
    # BUG: I think that this should not happen with the append
    deposition_summed = np.load(deposition_efficiency_file,allow_pickle=True)
    deposition_summed = np.append(deposition_summed[0],50000.5)
    # deposition_summed = deposition_summed[0]
    
    if len(deposition_summed) == 16:
        
        deposition_summed = np.insert(deposition_summed,0,0)

    original_energies_keV = np.array([0, 30, 40, 50 ,60, 70, 80 ,90 ,100 ,300 ,500 ,700, 900, 1000 ,2000 ,4000 ,6000])

    # Divide by the energy to get the photon count plus a factor 355000 for the original number of photons
    deposition_summed[1:] /= (original_energies_keV[1:]/355)
    
    print(deposition_summed.shape,original_energies_keV.shape)
    
    deposition_interpolated = np.interp(energies, original_energies_keV, deposition_summed)

    super_kernel = np.zeros([len(fluence),kernels.shape[1],kernels.shape[2]])


    for ii in range(kernels.shape[1]):
        for jj in range(kernels.shape[2]):
            
            super_kernel[:,ii,jj] = np.interp(np.array(energies), original_energies_keV, kernels[:,ii,jj])

    weights = fluence*deposition_interpolated
    weights = weights/sum(weights)

    normalized_kernel = super_kernel.copy()

    for ii in range(0,normalized_kernel.shape[0]):
        
        sum_kern = np.sum(super_kernel[ii,:,:])
        
        if sum_kern > 0:
            normalized_kernel[ii,:,:] = super_kernel[ii,:,:]/sum_kern

    new_kern = Kernel()

    # print((normalized_kernel.T*weights).shape)

    new_kern.kernel = normalized_kernel.T@weights


    return new_kern, kernels