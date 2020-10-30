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
import astropy.stats as stats

import tigre
try:
    import astra
except ImportError as error:
    # Output expected ImportErrors.
    print(error.__class__.__name__ + ": " + error.message)

import tigre.algorithms as algs
from scipy.signal import fftconvolve, find_peaks, butter,filtfilt
from scipy.optimize import minimize, curve_fit
from scipy.ndimage import gaussian_filter
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

mu_en_water = np.array([4.944, 0.5503, 0.1557  , 0.06947 , 0.04223 , 0.0319  , 0.027678, 0.02597 ,
    0.025434, 0.02546 , 0.03192 , 0.03299 , 0.032501, 0.031562,
    0.03103 , 0.02608 , 0.02066 , 0.01806 ])
mu_water = np.array([5.329, 0.8096, 0.3756  , 0.2683  , 0.2269  , 0.2059  , 0.19289 , 0.1837  ,
    0.176564, 0.1707  , 0.1186  , 0.09687 , 0.083614, 0.074411,
    0.07072 , 0.04942 , 0.03403 , 0.0277  ])
mu_woutcoherent_water = np.array([3.286E-01  , 2.395E-01   , 2.076E-01 , 1.920E-01  , 1.824E-01 , 1.755E-01  ,
    1.700E-01, 1.654E-01 , 1.180E-01 , 9.665E-02 , 8.351E-02, 7.434E-02,
    7.066E-02 , 4.940E-02 , 0.03402 , 0.0277  ])

# --------------------General purpose functions-------------------------#


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

def tigre2astra(phantom,geomet,angles,tile=False,init=False):

    tigre_shape = geomet.nVoxel
    
    # Create a geometry object
    vol_geom = astra.create_vol_geom(tigre_shape[1],tigre_shape[2])
    # Create the ratio of 1mm to 1 voxel
    ratio = tigre_shape[1]/geomet.sVoxel[0]
    # create the projection
    proj_geom = astra.create_proj_geom('fanflat', ratio*geomet.dDetector[0],
                                       geomet.nDetector[1], angles - np.pi/2,
                                       ratio*geomet.DSO,ratio*(geomet.DSD-geomet.DSO))
    # For CPU-based algorithms, a "projector" object specifies the projection
    # model used. In this case, we use the "strip" model.
    proj_id = astra.create_projector('strip_fanflat', proj_geom, vol_geom)
    
    if init:
        return proj_id, vol_geom
    
    if tile:
        sin_id, sinogram = astra.create_sino(phantom[0,:,:], proj_id)
        
        return np.tile(sinogram,[tigre_shape[0],1,1])/(1.6*(geomet.nDetector[1]/256)),sin_id, proj_id, vol_geom
    else:
        sinogram = np.zeros([tigre_shape[0],len(angles),geomet.nDetector[1]])
        
        sin_id = None
        
        for ii in range(tigre_shape[0]):
            
            if sin_id is not None:
                astra.data2d.delete(sin_id)
                
            sin_id, sinogram[ii,:,:] = astra.create_sino(phantom[ii,:,:], proj_id) # this is some sort of relation
            
        return sinogram/(1.6*(geomet.nDetector[1]/256)), sin_id, proj_id, vol_geom

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

# def get_noise(length):

#     if length == 512:

#         primary_large = np.load('/home/xcite/xpecgen/tests/primary_kernel_int_larger.npy')
#         noise = np.load('/home/xcite/xpecgen/tests/noise_projections.npy')
#         primary_512 = (primary_large[:,::2,:,:] + primary_large[:,1::2,:,:])/2
#         primary = np.tile(primary_512,[1,1,2,1])

#         return primary, noise
#     elif length == 1024:
#         primary_large = np.load('/home/xcite/xpecgen/tests/primary_kernel_int_larger.npy')
#         noise_512 = np.load('/home/xcite/xpecgen/tests/noise_projections.npy')
#         primary_256 = (primary_large[:,::2,:,:] + primary_large[:,1::2,:,:])/2
#         primary_512 = np.tile(primary_256,[1,1,2,1])

#         x = np.linspace(0,512,512,endpoint=False)
#         xp = np.linspace(0,512,1024,endpoint=False)

#         return np.interp(x,xp,primary_512),np.interp(x,xp,noise_512)

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

    def __init__(self,spectrum,detector,Verbose = False):

        dump_files = os.path.join(
            data_path, "Detectors", detector, '*phsp.npy')
        self.deposition_efficiency_file = os.path.join(
            data_path, "Detectors", detector, 'EnergyDeposition.npy')

        files = sorted(glob(dump_files))

        for ii, file in enumerate(files):
            
            if ii == 0:
                
                # Make the first entry zeros
                first_kernel = np.load(file)
                kernels = np.zeros([len(files)+1,first_kernel.shape[0],first_kernel.shape[1]]) 
                
            kernels[ii+1] = np.load(file)
                
        fluence = spectrum.y/sum(spectrum.y)
        
        deposition_summed = np.load(self.deposition_efficiency_file,allow_pickle=True)
        deposition_summed = np.insert(deposition_summed[0],0,0)
        
        if len(deposition_summed) == 16:
            deposition_summed = np.insert(deposition_summed,0,0)

        if len(deposition_summed) == 19:
            original_energies_keV = np.array([0, 10 ,20, 30, 40, 50 ,60, 70, 80 ,90 ,100 ,300 ,500 ,700, 900, 1000 ,2000 ,4000 ,6000])
        else:
            original_energies_keV = np.array([0, 30, 40, 50 ,60, 70, 80 ,90 ,100 ,300 ,500 ,700, 900, 1000 ,2000 ,4000 ,6000])

        # Divide by the energy to get the photon count plus a factor 355000 for the original number of photons
        deposition_summed[1:] /= (original_energies_keV[1:]/355)
        deposition_interpolated = np.interp(spectrum.x, original_energies_keV, deposition_summed)
        super_kernel = np.zeros([len(fluence),kernels.shape[1],kernels.shape[2]])


        print(kernels.shape,len(deposition_summed))
        for ii in range(kernels.shape[1]):
            for jj in range(kernels.shape[2]):
                
                super_kernel[:,ii,jj] = np.interp(np.array(spectrum.x), original_energies_keV, kernels[:,ii,jj])
        
        # normalized_kernel = super_kernel.copy()

        # for ii in range(0,normalized_kernel.shape[0]):
            
        #     sum_kern = np.sum(super_kernel[ii,:,:])
            
        #     if sum_kern > 0:
        #         normalized_kernel[ii,:,:] = super_kernel[ii,:,:]/sum_kern


        weights = fluence*deposition_interpolated
        self.weights = weights
        weights = weights/sum(weights)

        self.kernels = kernels
        self.kernel = super_kernel.T@weights
        self.pitch = int(detector[-14:-11])/1000  #mm

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

    def get_plot_mtf_real(self, place, show_mesh=True, prepare_format=True,label=''):

        h,w = 1024*4,1024 #Wouldn't change tbh for building lsf
        step = 16*2 #Wouldn't change tbh for building lsf
        pitch = self.pitch #mm 
        angle = 2.3 #deg
        lsf_width = 0.3 #mm Wouldn't change tbh

        # --- Make a high res line ---

        high_res = np.zeros([h,w])
        Y, X = np.mgrid[:h, :w]
        dist_from_line = np.abs((X - high_res.shape[1]/2) + Y*np.tan(angle*np.pi/180))
        # The MTF is from a 0.3 mm pixel times the angle times 16 since it will be averaged over 32 pix
        num_pix = lsf_width*1/cos(angle*np.pi/180)/pitch*16
        high_res[dist_from_line < num_pix] = 1

        # --- Average to make low res line ---
        # Ugly sorry
        low_res = np.array([[np.mean(high_res[ii:ii+step,jj:jj+step]) 
                            for ii in range(0,h,step)] for jj in range(0,w,step)]).T

        # --- Convlolve with the kernel ---
        lsf_image = fftconvolve(low_res,self.kernel/np.sum(self.kernel),mode='same')

        # --- Pad and presample ---
        pad_len = int((512 - lsf_image.shape[1])/2)
        lsf_image =  np.pad(lsf_image,((0,0),(pad_len,pad_len)),mode='constant')
        Y, X = np.mgrid[:lsf_image.shape[0], :lsf_image.shape[1]]
        center = int(lsf_image.shape[1]/2)
        # pitch needs to convert to cm from mm
        dist_from_line = (X + Y*np.tan(angle*np.pi/180) - center + 0.5)*pitch/10

        # --- Crop the convolved edges ---
        inds = np.argsort(dist_from_line[10:-10,:].flatten())
        line = dist_from_line[10:-10,:].flatten()[inds]
        lsf = lsf_image[10:-10,:].flatten()[inds]
        n,bins = np.histogram(line,818,weights=lsf,density=True)

        # if plot_stuff:
        #     plt.figure()
        #     plt.plot(bins[1:],n/(np.sum(n)),'x-',linewidth= 1.1,markersize=5,color='cornflowerblue')
        #     plt.title('LSF')
        #     plt.legend(['fastCAT','geant4'])
        #     plt.xlabel('[mm]')
        #     plt.ylabel('Normalized Amplitude')
        #     plt.xlim([-0.5,0.5])
        #     plt.savefig('LSF_good')

        # --- fft to get mtf ---
        n,bins = np.histogram(line,818,weights=lsf,density=True)
        mtf = np.absolute(np.fft.fft(n))
        mtf_final = np.fft.fftshift(mtf)
        N = len(mtf)
        T = np.mean(np.diff(bins))
        xf = np.linspace(0.0, 1.0/(2.0*T), int((N-1)/2))
        mm = np.argmax(mtf_final)


        place.plot(xf/10,mtf_final[mm+1:]/mtf_final[mm+1],'--',linewidth= 1.1,markersize=2,label=label)
        place.set_xlim((0,1/(2*pitch)))
        place.set_xlabel('Spatial Frequency [1/mm]')
        place.set_ylabel('MTF')
        place.legend()
        place.grid(True)

        self.mtf = mtf_final[mm+1:]/mtf_final[mm+1]
        self.freq = xf/10

    def add_focal_spot(self,fs_size_in_pix):

        self.kernel = gaussian_filter(self.kernel,sigma=fs_size_in_pix)

class Phantom:

    def __init__(self):
        pass

    def return_projs(self,kernel,spectra,angles,nphoton = None,
                    mgy = 0.,return_dose=False,det_on=True,scat_on=True,tigre_works = True):
        '''
        The main function for returning the projections
        '''

        # --- Making the phantom maps between G4 attenuation coeffs and ints in phantom ---

        # Don't want to look for zeros
        useful_phantom = self.phantom != 0
        # These are what I used in the Monte Carlo
        deposition = np.load(kernel.deposition_efficiency_file,allow_pickle=True)
        
        # csi has two extra kv energies
        if len(deposition[0]) == 18:
            original_energies_keV = np.array([10,20,30, 40, 50 ,60, 70, 80 ,90 ,100 ,300 ,500 ,700, 900, 1000 ,2000 ,4000 ,6000])
            mu_en_water2 = mu_en_water
            mu_water2 = mu_water
        else:
            original_energies_keV = np.array([30, 40, 50 ,60, 70, 80 ,90 ,100 ,300 ,500 ,700, 900, 1000 ,2000 ,4000 ,6000])
            mu_en_water2 = mu_en_water[2:]
            mu_water2 = mu_water[2:]
        # Loading the file from the monte carlo
        # This is a scaling factor that I found to work to convert energy deposition to photon probability eta
        deposition_summed = deposition[0]/(original_energies_keV/1000)/1000000
        # The index of the different materials
        masks = np.zeros([len(self.phan_map)-1,useful_phantom.shape[0],useful_phantom.shape[1],useful_phantom.shape[2]])
        mapping_functions = []
        # Get the mapping functions for the different tissues to reconstruct the phantom by energy
        for ii in range(1,len(self.phan_map)):       
            mapping_functions.append(get_mu(self.phan_map[ii].split(':')[0]))
            masks[ii-1] = self.phantom == ii
        phantom2 = self.phantom.copy().astype(np.float32) # Tigre only works with float32

        self.angles = angles

        # --- Ray tracing step ---        
        proj = []
        doses = []
        # we assume tigre works
        tile = True

        if not tigre_works:
            self.proj_id, self.vol_geom = tigre2astra(phantom2,self.geomet,angles,init=True)

        calc_projs = True

        # Don't bother double calcing the projs if we already have them from a previous run
        if hasattr(self, 'proj') and hasattr(self, 'n_energies_old'):
            if self.angles.shape[0] in self.proj.shape:
                if len(original_energies_keV) == self.n_energies_old:
                    calc_projs = False

        self.n_energies_old = len(original_energies_keV)

        if calc_projs:
            for jj, energy in enumerate(original_energies_keV):
                # Change the phantom values
                for ii in range(0,len(self.phan_map)-1):
                    phantom2[masks[ii].astype(bool)] = mapping_functions[ii](energy)
                
                if tigre_works: # resort to astra if tigre doesn't work
                    try:
                        proj.append(np.squeeze(tigre.Ax(phantom2,self.geomet,angles)))
                    except Exception:
                        print("WARNING: Tigre GPU not working. Switching to Astra CPU")

                        sinogram, self.sin_id, self.proj_id, self.vol_geom = tigre2astra(phantom2,self.geomet,angles,tile=True)
                        proj.append(sinogram.transpose([1,0,2]))
                        tigre_works=False
                else:
                    if tile:
                        sin_id, sinogram = astra.create_sino(np.fliplr(phantom2[0,:,:]), self.proj_id)
                        
                        proj.append(np.tile(sinogram,[phantom2.shape[0],1,1]).transpose([1,0,2])/(1.6*(geomet.nDetector[1]/256)))
                    else:
                        sinogram = np.zeros([phantom2.shape[0],len(angles),geomet.nDetector[1]])
                        
                        sin_id = None
                        
                        for kk in range(tigre_shape[0]):
                            
                            if sin_id is not None:
                                astra.data2d.delete(sin_id)
                                
                            sin_id, sinogram[kk,:,:] = astra.create_sino(np.fliplr(phantom2[kk,:,:]), self.proj_id)
                        
                        proj.append(sinogram.transpose([1,0,2])/(1.6*(self.geomet.nDetector[1]/256)))

                    astra.data2d.delete(sin_id)
                # Calculate a dose contribution by dividing by 10 since tigre has projections that are a little odd
                doses.append((energy)*(1-np.exp(-(proj[-1][0]*.997)/10))*mu_en_water2[jj]/mu_water2[jj])

        if calc_projs:
            self.raw_proj = proj
            self.raw_doses = doses
        
        # --- Factoring in the fluence and the energy deposition ---
        # Binning to get the fluence per energy
        large_energies = np.linspace(0,6000,3001)
        fluence_large = np.interp(large_energies,np.array(spectra.x), spectra.y)
        fluence_small = np.zeros(len(original_energies_keV))
        # Still binning
        for ii, val in enumerate(large_energies):   
            index = np.argmin(np.abs(original_energies_keV-val))
            fluence_small[index] += fluence_large[ii]       
        # Normalize
        fluence_small /= np.sum(fluence_small)
        fluence_norm = spectra.y/np.sum(spectra.y)

        # if det_on:
        weights_small = fluence_small*deposition_summed
        # else:
        #     weights_small = fluence_small
        
        # Need to make sure that the attenuations aren't janky for recon
        weights_small /= np.sum(weights_small)
        # This is the line to uncomment to run the working code for dose_comparison.ipynb
        if return_dose:
            return np.mean(np.mean(self.raw_doses,1),1), spectra.y
        
        # --- Dose calculation ---
        # Sum over the image dimesions to get the energy intensity and multiply by fluence TODO: what is this number?
        def get_dose_nphoton(nphot):
            return nphot/2e7

        def get_dose_mgy(mgy,doses,fluence_small):
            nphoton = mgy/(get_dose_per_photon(doses,fluence_small)*(1.6021766e-13)*1000)
            return get_dose_nphoton(nphoton)

        def get_dose_per_photon(doses,fluence_small):
            # linear fit of the data
            pp = np.array([0.88759883, 0.01035186])
            return ((np.mean(np.mean(doses,1),1)/1000)@(fluence_small))*pp[0] + pp[1]

        ratio = None

        # Dose in micro grays
        if mgy != 0.0:
            ratio = get_dose_mgy(mgy,self.raw_doses,fluence_small)
        elif nphoton is not None:
            ratio = get_dose_nphoton(nphoton)
        
        # --- Noise and Scatter Calculation ---
        # Now I interpolate deposition and get the average photons reaching the detector
        deposition_long = np.interp(spectra.x,original_energies_keV,deposition_summed) ## !! big change
        nphotons_at_energy = fluence_norm*deposition_long
        nphotons_av = np.sum(nphotons_at_energy)

        print('The ratio compared to the mc', ratio,' number of photons', nphotons_av)

        # -------- Scatter Correction -----------
        scatter = np.load(os.path.join(data_path,'scatter','scatter.npy'))
        coh_scatter = np.load(os.path.join(data_path,'scatter','coherent_scatter.npy'))

        dist = np.linspace(-256*0.0784 - 0.0392,256*0.0784 - 0.0392, 512)

        def func(x, a, b):
            return ((-(152/(np.sqrt(x**2 + 152**2)))**a)*b)

        mc_scatter = np.zeros(scatter.shape)

        for jj in range(16):

            popt, popc = curve_fit(func,dist,scatter[:,jj],[10,scatter[256,jj]])
            mc_scatter[:,jj] = func(dist, *popt)

        print(mc_scatter.shape)
        print((mc_scatter[:,0].shape))

        if len(original_energies_keV) == 18:
            mc_scatter = np.vstack((mc_scatter[:,0].T,mc_scatter[:,0].T,mc_scatter.T))
            # mc_scatter = np.vstack((mc_scatter[:,0].T,mc_scatter))
            mc_scatter = mc_scatter.T

            coh_scatter = np.vstack((coh_scatter[:,0].T,coh_scatter[:,0].T,coh_scatter.T))
            # coh_scatter = np.vstack((coh_scatter[:,0].T,coh_scatter))
            coh_scatter = coh_scatter.T

        factor = (152/(np.sqrt(dist**2 + 152**2)))**3
        flood_summed = factor*660 
        scatter = 2.15*(mc_scatter + coh_scatter)
        # log(i/i_0) to get back to intensity
        raw = (np.exp(-np.array(self.raw_proj)/10)*(flood_summed)).T

        # Add the already weighted noise
        if scat_on:
            raw_weighted = raw.transpose([2,1,0,3]) + scatter
        else:
            raw_weighted = raw.transpose([2,1,0,3])

        # Normalize the kernel
        kernel_norm = kernel.kernel/np.sum(kernel.kernel)
            
        # add the poisson noise
        if ratio is not None:
            
            # if det_on:
            adjusted_ratio = ratio*nphotons_av
            # else:
            #     adjusted_ratio = ratio
            
            raw_weighted = np.random.poisson(lam=raw_weighted*adjusted_ratio)/adjusted_ratio
        
        filtered = raw_weighted.copy()

        if det_on: # if the detector is to be simulated

            for ii in range(len(angles)):
                for jj in range(len(original_energies_keV)):

                    kernel.kernels[jj+1] /= np.sum(kernel.kernels[jj+1])

                    filtered[ii,:,:,jj] = fftconvolve(raw_weighted[ii,:,:,jj],kernel.kernels[jj+1], mode = 'same')

        filtered = filtered @ weights_small

        self.proj = -10*np.log(filtered/(flood_summed))

    def plot_projs(self,fig):

        subfig1 = fig.add_subplot(121)
        subfig2 = fig.add_subplot(122)

        tracker = IndexTracker(
            subfig1, self.proj.transpose([1,2,0]))
        fig.canvas.mpl_connect(
            'scroll_event', tracker.onscroll)
        tracker2 = IndexTracker(
            subfig2, self.proj.transpose([0,2,1]))
        fig.canvas.mpl_connect(
            'scroll_event', tracker2.onscroll)
        fig.tight_layout()

    def reconstruct(self,algo,filt='hamming'):
        
        if algo == 'FDK':
            try:
                self.img = tigre.algorithms.FDK(self.proj, self.geomet, self.angles,filter=filt)
            except Exception:
                print('WARNING: Tigre failed during recon using Astra')
                self.img = self.astra_recon(self.proj.transpose([1,0,2]))

    def astra_recon(self,projs,algo ='CGLS',niter=10):

        sinogram, sin_id, proj_id, vol_geom = tigre2astra(self.phantom,self.geomet,self.angles,tile=True)
    
        rec_id = astra.data2d.create('-vol', vol_geom)

        cfg = astra.astra_dict('CGLS')
        
        cfg['ReconstructionDataId'] = rec_id
        cfg['ProjectionDataId'] = sin_id
        cfg['ProjectorId'] = proj_id
        
        # Create the algorithm object from the configuration structure
        alg_id = astra.algorithm.create(cfg)

        recon = []

        for ii in range(projs.T.shape[2]):

            # Available algorithms:
            # ART, SART, SIRT, CGLS, FBP
            astra.data2d.store(sin_id,projs[ii,:,:]*(1.6*(self.geomet.nDetector[1]/256)))

            # Run 20 iterations of the algorithm
            # This will have a runtime in the order of 10 seconds.
            astra.algorithm.run(alg_id,niter)

            # Get the result
            rec = astra.data2d.get(rec_id)

            recon.append(rec)

        astra.algorithm.delete(alg_id)
        astra.data2d.delete(rec_id)
        astra.data2d.delete(sin_id)
        astra.projector.delete(proj_id)
            
        return np.array(recon)
        
class Catphan_515(Phantom):

    def __init__(self): #,det):
        self.phantom = np.load(os.path.join(data_path,'phantoms','catphan_low_contrast_512_8cm.npy'))
        self.geomet = tigre.geometry_default(high_quality=False,nVoxel=self.phantom.shape)
        self.geomet.DSD = 1520 #1500 + 20 for det casing
        self.geomet.nDetector = np.array([64,512])
        self.geomet.dDetector = np.array([0.784, 0.784])#det.pitch, det.pitch]) #TODO: Change this to get phantom

        # I think I can get away with this
        self.geomet.sDetector = self.geomet.dDetector * self.geomet.nDetector    

        self.geomet.sVoxel = np.array((160, 160, 160)) 
        self.geomet.dVoxel = self.geomet.sVoxel/self.geomet.nVoxel
        self.phan_map = ['air','water','G4_LUNG_ICRP',"G4_BONE_COMPACT_ICRU","G4_BONE_CORTICAL_ICRP","G4_ADIPOSE_TISSUE_ICRP","G4_BRAIN_ICRP","G4_B-100_BONE"] 

    def analyse_515(self,recon_slice,place = None,run_name = ''):

        def create_mask(shape):

            im = np.zeros(shape)
            ii = 1
            
            offset = 0.1
            
            first_radius = 5 - offset
            second_radius = 2.5 - offset
            
            correction = -2*np.pi/180
            # CTMAT(x) formel=H2O dichte=x
            LEN = 100

            A0  = 87.7082*np.pi/180 + correction
            A1 = 108.3346*np.pi/180 + correction
            A2 = 126.6693*np.pi/180 + correction
            A3 = 142.7121*np.pi/180 + correction
            A4 = 156.4631*np.pi/180 + correction
            A5 = 167.9223*np.pi/180 + correction
            A6 = 177.0896*np.pi/180 + correction
            A7 = 183.9651*np.pi/180 + correction
            A8 = 188.5487*np.pi/180 + correction

            B0 = 110.6265*np.pi/180 + correction
            B1 = 142.7121*np.pi/180 + correction
            B2 = 165.6304*np.pi/180 + correction
            B3 = 179.3814*np.pi/180 + correction

            tad = 0.2

            # Phantom 
            # ++++ module body ++++++++++++++++++++++++++++++++++++++++++++++++++ */                        
            create_circular_mask(x= 0.000,  y= 0.000,  r=-tad + 2, index = ii, image = im)

            ii += 1

            # ++++ supra-slice 1.0% targets +++++++++++++++++++++++++++++++++++++++ */
            create_circular_mask(x= first_radius*cos(A0),  y= first_radius*sin(A0),  r=-tad + 0.75, index = ii, image = im); ii += 1
            create_circular_mask(x= first_radius*cos(A1),  y= first_radius*sin(A1),  r=-tad + 0.45, index = ii, image = im); ii += 1
            create_circular_mask(x= first_radius*cos(A2),  y= first_radius*sin(A2),  r=-tad + 0.40, index = ii, image = im); ii += 1
            create_circular_mask(x= first_radius*cos(A3),  y= first_radius*sin(A3),  r=-tad + 0.35, index = ii, image = im); ii += 1
            create_circular_mask(x= first_radius*cos(A4),  y= first_radius*sin(A4),  r=-tad + 0.30, index = ii, image = im); ii += 1
            create_circular_mask(x= first_radius*cos(A5),  y= first_radius*sin(A5),  r=-tad + 0.25, index = ii, image = im); ii += 1
            create_circular_mask(x= first_radius*cos(A6),  y= first_radius*sin(A6),  r=-tad + 0.20, index = ii, image = im); ii += 1
            create_circular_mask(x= first_radius*cos(A7),  y= first_radius*sin(A7),  r=-tad + 0.15, index = ii, image = im); ii += 1
            create_circular_mask(x= first_radius*cos(A8),  y= first_radius*sin(A8),  r=-tad + 0.10, index = ii, image = im); ii += 1


            # ++++ supra-slice 0.3% targets +++++++++++++++++++++++++++++++++++++++ */
            create_circular_mask(x= first_radius*cos(A0+2/3*np.pi),  y= first_radius*sin(A0+2/3*np.pi),  r=-tad + 0.75, index = ii, image = im); ii += 1
            create_circular_mask(x= first_radius*cos(A1+2/3*np.pi),  y= first_radius*sin(A1+2/3*np.pi),  r=-tad + 0.45, index = ii, image = im); ii += 1
            create_circular_mask(x= first_radius*cos(A2+2/3*np.pi),  y= first_radius*sin(A2+2/3*np.pi),  r=-tad + 0.40, index = ii, image = im); ii += 1
            create_circular_mask(x= first_radius*cos(A3+2/3*np.pi),  y= first_radius*sin(A3+2/3*np.pi),  r=-tad + 0.35, index = ii, image = im); ii += 1
            create_circular_mask(x= first_radius*cos(A4+2/3*np.pi),  y= first_radius*sin(A4+2/3*np.pi),  r=-tad + 0.30, index = ii, image = im); ii += 1
            create_circular_mask(x= first_radius*cos(A5+2/3*np.pi),  y= first_radius*sin(A5+2/3*np.pi),  r=-tad + 0.25, index = ii, image = im); ii += 1
            create_circular_mask(x= first_radius*cos(A6+2/3*np.pi),  y= first_radius*sin(A6+2/3*np.pi),  r=-tad + 0.20, index = ii, image = im); ii += 1
            create_circular_mask(x= first_radius*cos(A7+2/3*np.pi),  y= first_radius*sin(A7+2/3*np.pi),  r=-tad + 0.15, index = ii, image = im); ii += 1
            create_circular_mask(x= first_radius*cos(A8+2/3*np.pi),  y= first_radius*sin(A8+2/3*np.pi),  r=-tad + 0.10, index = ii, image = im); ii += 1


            # ++++ supra-slice 0.5% targets +++++++++++++++++++++++++++++++++++++++ */
            create_circular_mask(x= first_radius*cos(A0+4/3*np.pi),  y= first_radius*sin(A0+4/3*np.pi),  r=-tad + 0.75, index = ii, image = im); ii += 1
            create_circular_mask(x= first_radius*cos(A1+4/3*np.pi),  y= first_radius*sin(A1+4/3*np.pi),  r=-tad + 0.45, index = ii, image = im); ii += 1
            create_circular_mask(x= first_radius*cos(A2+4/3*np.pi),  y= first_radius*sin(A2+4/3*np.pi),  r=-tad + 0.40, index = ii, image = im); ii += 1
            create_circular_mask(x= first_radius*cos(A3+4/3*np.pi),  y= first_radius*sin(A3+4/3*np.pi),  r=-tad + 0.35, index = ii, image = im); ii += 1
            create_circular_mask(x= first_radius*cos(A4+4/3*np.pi),  y= first_radius*sin(A4+4/3*np.pi),  r=-tad + 0.30, index = ii, image = im); ii += 1
            create_circular_mask(x= first_radius*cos(A5+4/3*np.pi),  y= first_radius*sin(A5+4/3*np.pi),  r=-tad + 0.25, index = ii, image = im); ii += 1
            create_circular_mask(x= first_radius*cos(A6+4/3*np.pi),  y= first_radius*sin(A6+4/3*np.pi),  r=-tad + 0.20, index = ii, image = im); ii += 1
            create_circular_mask(x= first_radius*cos(A7+4/3*np.pi),  y= first_radius*sin(A7+4/3*np.pi),  r=-tad + 0.15, index = ii, image = im); ii += 1
            create_circular_mask(x= first_radius*cos(A8+4/3*np.pi),  y= first_radius*sin(A8+4/3*np.pi),  r=-tad + 0.10, index = ii, image = im); ii += 1

            # ++++ subslice 1.0% targets 7mm long +++++++++++++++++++++++++++++++++ */
            create_circular_mask(x= second_radius*cos(B0),  y= second_radius*sin(B0),  r=-tad + 0.45, index = ii, image = im); ii += 1
            create_circular_mask(x= second_radius*cos(B1),  y= second_radius*sin(B1),  r=-tad + 0.35, index = ii, image = im); ii += 1
            create_circular_mask(x= second_radius*cos(B2),  y= second_radius*sin(B2),  r=-tad + 0.25, index = ii, image = im); ii += 1
            create_circular_mask(x= second_radius*cos(B3),  y= second_radius*sin(B3),  r=-tad + 0.15, index = ii, image = im); ii += 1


            # ++++ subslice 1.0% targets 3mm long +++++++++++++++++++++++++++++++++ */
            create_circular_mask(x= second_radius*cos(B0+2/3*np.pi) ,y= second_radius*sin(B0+2/3*np.pi),  r=-tad + 0.45  , index = ii, image = im); ii += 1
            create_circular_mask(x= second_radius*cos(B1+2/3*np.pi) ,y= second_radius*sin(B1+2/3*np.pi),  r=-tad + 0.35  , index = ii, image = im); ii += 1
            create_circular_mask(x= second_radius*cos(B2+2/3*np.pi) ,y= second_radius*sin(B2+2/3*np.pi),  r=-tad + 0.25  , index = ii, image = im); ii += 1
            create_circular_mask(x= second_radius*cos(B3+2/3*np.pi) ,y= second_radius*sin(B3+2/3*np.pi),  r=-tad + 0.15  , index = ii, image = im); ii += 1


            # ++++ subslice 1.0% targets 5mm long +++++++++++++++++++++++++++++++++ */
            create_circular_mask(x= second_radius*cos(B0+4/3*np.pi) ,y= second_radius*sin(B0+4/3*np.pi),  r=-tad + 0.45  , index = ii, image = im); ii += 1
            create_circular_mask(x= second_radius*cos(B1+4/3*np.pi) ,y= second_radius*sin(B1+4/3*np.pi),  r=-tad + 0.35  , index = ii, image = im); ii += 1
            create_circular_mask(x= second_radius*cos(B2+4/3*np.pi) ,y= second_radius*sin(B2+4/3*np.pi),  r=-tad + 0.25  , index = ii, image = im); ii += 1
            create_circular_mask(x= second_radius*cos(B3+4/3*np.pi) ,y= second_radius*sin(B3+4/3*np.pi),  r=-tad + 0.15  , index = ii, image = im); ii += 1

            return im

        def create_mask_multi(shape):

            im = np.zeros(shape)
            ii = 1
            
            correction = 0
            first_radius = 5
            second_radius = 2.5

            # CTMAT(x) formel=H2O dichte=x
            LEN = 100

            A0  = 87.7082*np.pi/180 + correction
            A1 = 108.3346*np.pi/180 + correction
            A2 = 126.6693*np.pi/180 + correction
            A3 = 142.7121*np.pi/180 + correction
            A4 = 156.4631*np.pi/180 + correction
            A5 = 167.9223*np.pi/180 + correction
            A6 = 177.0896*np.pi/180 + correction
            A7 = 183.9651*np.pi/180 + correction
            A8 = 188.5487*np.pi/180 + correction
            B0 = 110.6265*np.pi/180 + correction
            B1 = 142.7121*np.pi/180 + correction
            B2 = 165.6304*np.pi/180 + correction
            B3 = 179.3814*np.pi/180 + correction

            # Phantom 
            # ++++ module body ++++++++++++++++++++++++++++++++++++++++++++++++++ */                        
            create_circular_mask(x= 0.000,  y= 0.000,  r=1.0, index = ii, image = im)

            ii += 1

            tad = 0.5
            # ++++ supra-slice 1.0% targets +++++++++++++++++++++++++++++++++++++++ */

            create_circular_mask(x= first_radius*cos(A0),  y= first_radius*sin(A0),  r=0.75 - tad, index = ii, image = im); ii += 1
            create_circular_mask(x= first_radius*cos(A0+2/3*np.pi),  y= first_radius*sin(A0+2/3*np.pi),  r=0.75 - tad, index = ii, image = im); ii += 1
            create_circular_mask(x= first_radius*cos(A0+4/3*np.pi),  y= first_radius*sin(A0+4/3*np.pi),  r=0.75 - tad, index = ii, image = im); ii += 1

            create_circular_mask(x= second_radius*cos(B0),  y= second_radius*sin(B0),  r=0.45 - tad, index = ii, image = im); ii += 1
            create_circular_mask(x= second_radius*cos(B0+2/3*np.pi),  y= second_radius*sin(B0+2/3*np.pi),  r=0.45 - tad  , index = ii, image = im); ii += 1
            create_circular_mask(x= second_radius*cos(B0+4/3*np.pi),  y= second_radius*sin(B0+4/3*np.pi),  r=0.45 - tad  , index = ii, image = im); ii += 1       
            return im

        def create_circular_mask(x, y, r, index, image):
        
            h,w = image.shape
            
            center = [x*int(w/2)/8 + int(w/2),y*int(h/2)/8 + int(h/2)]

            if center is None: # use the middle of the image
                center = (int(w/2), int(h/2))
            if r is None: # use the smallest distance between the center and image walls
                radius = min(center[0], center[1], w-center[0], h-center[1])

            Y, X = np.ogrid[:h, :w]
            dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

            mask = dist_from_center <= r*int(w/2)/8
            
            
            image[mask] = index

        im = create_mask(recon_slice.shape)

        contrast = []
        noise = []
        cnr = []
        ci = []

        ii = 1

        ref_mean = np.mean(recon_slice[im == ii])
        ref_std = np.std(recon_slice[im == ii])

        # for ii in range(2,int(np.max(im)+1)):
            
        #     contrast.append(np.abs(np.mean(recon_slice[im == ii])- ref_mean))
        #     noise.append(np.std(recon_slice[im == ii]))
            
        #     cnr.append(contrast[-1]/(np.sqrt(noise[-1]**2)))

        for ii in range(2,int(np.max(im)+1)):
            
            nsample = len(recon_slice[im == ii])
            
            if nsample > 2:
                
                noise.append(np.std(recon_slice[im == ii]))
                
                booted = np.abs(stats.bootstrap(recon_slice[im == ii],100,samples=int(nsample/5),bootfunc=np.mean) - ref_mean)

                ci.append(np.std(booted))
                contrast.append(np.mean(booted))

                cnr.append(contrast[-1]/(np.sqrt(noise[-1]**2)))
                      
        ci_v = [2*(ci[ii]/ref_mean)*100 for ii in range(len(ci))]
            
        rs = np.linspace(0.1,0.45,8)

        inds_i_want = [0,6,12,18,21,24]
        ww = 0.085

        shorts = ['Lung','Compact Bone','Cortical Bone','Adipose','Brain','B-100']

        contrasts_i_want = np.array([(contrast[ii]/ref_mean)*100 for ii in range(len(contrast))])[inds_i_want]

        place[0].errorbar(np.arange(len(inds_i_want)),contrasts_i_want
                                ,capsize = ww+1.5, yerr=np.array(ci_v)[inds_i_want],fmt='x',label=run_name)
        place[0].set_xticks(range(len(inds_i_want))) 
        place[0].set_xticklabels(shorts, fontsize=12, rotation = 75)
        place[0].set_ylabel('% Contrast')
        place[0].set_title('Contrast')
        place[0].legend()

        place[1].errorbar(np.arange(len(inds_i_want)),np.array(cnr)[inds_i_want],capsize = ww+1.5,
                            yerr=np.array(cnr)[inds_i_want]*(np.array(ci_v)[inds_i_want]/contrasts_i_want),fmt='x')
        place[1].set_xticks(range(len(inds_i_want))) 
        place[1].set_xticklabels(shorts, fontsize=12, rotation = 75)
        place[1].set_ylabel('CNR')
        place[1].set_title('Contrast to Noise')

        return_contrast = True

        if return_contrast:
            return rs, [(contrast[ii]/ref_mean)*100 for ii in range(len(contrast))],ci_v, cnr,np.array(cnr)[inds_i_want]*(np.array(ci_v)[inds_i_want]/contrasts_i_want)

class Catphan_MTF(Phantom):

    def __init__(self):
        self.phantom = np.load(os.path.join(data_path,'phantoms/MTF_phantom_1024.npy'))
        self.geomet = tigre.geometry_default(high_quality=False,nVoxel=self.phantom.shape)
        self.geomet.nDetector = np.array([64,512])
        self.geomet.dDetector = np.array([0.784, 0.784])
        self.phan_map = ['air','water',"G4_BONE_COMPACT_ICRU"] 
        self.geomet.DSD = 1500
        # I think I can get away with this
        self.geomet.sDetector = self.geomet.dDetector * self.geomet.nDetector    

        self.geomet.sVoxel = np.array((160, 160, 160)) 
        self.geomet.dVoxel = self.geomet.sVoxel/self.geomet.nVoxel 

    def analyse_515(self,slc,place,fmt='-'):

        chunk_x = 100
        chunk_y = 35

        signal = []
        standev = []

        def get_diff(smoothed_slice):
            

                peak_info = find_peaks(smoothed_slice,height=0.07,prominence=0.005)
                
                peaks = peak_info[0]
                valleys = np.unique(np.hstack([peak_info[1]['right_bases'],peak_info[1]['left_bases']]))
                
                inds = np.array(valleys < np.max(peaks)) * np.array(valleys > np.min(peaks))
                
                valleys = valleys[inds]
                peaks = peaks[:-1]
                
                diffs = []
                
                for peak, valley in zip(peaks, valleys):
                    
                    diff = smoothed_slice[peak] - smoothed_slice[valley]
                    
                    diffs.append(diff)
                
                return diffs

        start_x = 310
        start_y = 270

        b, a = butter(3, 1/7, btype='low', analog=False)

        smoothed_slice = filtfilt(b, a, np.mean(slc[start_y:start_y+chunk_y,start_x:start_x+chunk_x],0)) #np.convolve(np.mean(slc[start_y:start_y+chunk_y,start_x:start_x+chunk_x],0),10*[0.1],'same')

        signal.append(np.mean(get_diff(smoothed_slice)))
        standev.append(0)

        b, a = butter(3, 1/5, btype='low', analog=False)
        b2, a2 = butter(3, 1/3.5, btype='low', analog=False)
        b3, a3 = butter(3, 1/2, btype='low', analog=False)

        for start_y, freq in zip([400,520,650],[[b,a],[b2,a2],[b3,a3]]):
            for start_x in [690, 570, 430, 300]: 
                
                # Need this to correct against the artifacts
                # if start_y == 400:
                # b, a = butter(3, 1/freq, btype='low', analog=False)
                smoothed_slice = filtfilt(freq[0], freq[1], np.mean(slc[start_y:start_y+chunk_y,start_x:start_x+chunk_x],0))
                # else:
                #     smoothed_slice = np.mean(slc[start_y:start_y+chunk_y,start_x:start_x+chunk_x],0)
                
                diffs = get_diff(smoothed_slice)
                
                if len(diffs) != 0:
                    signal.append(np.mean(diffs))
                    standev.append(np.std(diffs))
                else:
                    signal.append(0)
                    standev.append(0)
                    break
                

        pitch = 0.015625

        lpmm = [1/(2*ii*pitch) for ii in range(12,0,-1)]

        lpmm.insert(0,1/(2*30*pitch))

        place[0].errorbar(lpmm[:len(signal)],signal/signal[0],yerr=standev/signal[0],fmt='kx')
        place[0].plot(lpmm[:len(signal)],signal/signal[0],fmt)
        place[0].set_xlabel('lp/cm')
        place[0].set_ylabel('MTF')

        return [signal/signal[0], standev/signal[0]]

class XCAT(Phantom):

    def __init__(self):
        self.phantom = np.load(os.path.join(data_path,'phantoms','ct_scan_smaller.npy'))
        self.geomet = tigre.geometry_default(high_quality=False,nVoxel=self.phantom.shape)
        self.geomet.nDetector = np.array([512,512])
        self.geomet.dDetector = np.array([0.784, 0.784])
        self.phan_map = ['air','lung','adipose','adipose','water','water','G4_LUNG_ICRP','tissue4','testis','brain','tissue','tissue4','testis','brain',
            'breast','muscle','G4_MUSCLE_SKELETAL_ICRP','G4_MUSCLE_STRIATED_ICRU','G4_SKIN_ICRP','G4_TISSUE-PROPANE',
            'G4_TISSUE-METHANE','G4_TISSUE_SOFT_ICRP','G4_TISSUE_SOFT_ICRU-4','G4_BLOOD_ICRP','G4_BODY','G4_BONE_COMPACT_ICRU',
            'G4_BONE_CORTICAL_ICRP']

        self.geomet.DSD = 1500
        # I think I can get away with this
        self.geomet.sDetector = self.geomet.dDetector * self.geomet.nDetector    

        self.geomet.sVoxel = np.array((160, 160, 160)) 
        self.geomet.dVoxel = self.geomet.sVoxel/self.geomet.nVoxel 

    def analyse_515(self,slc,place,fmt='-'):

        pass

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

    def load(self, spectrum_file):

        energies = []
        fluence = []

        with open(os.path.join(data_path, "MV_spectra", f'{spectrum_file}.txt')) as f:
            for line in f:
                energies.append(float(line.split()[0]))
                fluence.append(float(line.split()[1]))

        # Check if MV

        self.x = np.array(energies)*1000  # to keV
        self.y = np.array(fluence)

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
    parser.add_argument('--e_0', metavar='E0', type=float, default=100.,
                        help='Electron kinetic energy in keV')
    parser.add_argument('--theta', metavar='theta', type=float, default=12,
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

    parser.add_argument('-s', metavar='spectrum', type=str,
                        help="Spectrum to load")

    parser.add_argument('-d', metavar='detector', type=str,
                        help="Detector to use")

    parser.add_argument('--fs', metavar='fs', type=int, default=0,
                        help="The size of the focal spot")

    parser.add_argument('-phan', metavar='phantom', type=str,
                        help="The phantom to select")

    parser.add_argument('--nviews', metavar='nviews', type=int, default=180,
                        help="The phantom to select")

    # parser.add_argument('-nviews', metavar='nviews', type=int, default=180,
    #                     help="The phantom to select")

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

    if args.s is None:
        s = calculate_spectrum_mesh(args.e_0, args.theta, mesh, phi=args.phi, epsrel=args.epsrel, monitor=monitor, z=args.z)
        x2, y2 = s.get_points()
    else:
        s = Spectrum()
        s.load(args.s)
        kernel = Kernel(s,args.d)
        
        dispatcher={'Catphan_515':Catphan_515,
                    'Catphan_MTF':Catphan_MTF}
        try:
            function=dispatcher[args.phan]
        except KeyError:
            raise ValueError('Invalid phantom module name')
        phantom = function()

        phantom.return_projs(kernel,s,np.linspace(0,np.pi*2,args.nviews))

        phantom.reconstruct('FDK')

        np.save(args.output,phantom.img)

    # if args.output is None:
    #     [print("%.6g, %.6g" % (x, y)) for x, y in zip(x2, y2)]
    # elif ext == "csv":
    #     s.export_csv(args.output)
    # elif ext == "xlsx":
    #     s.export_xlsx(args.output)
    # elif ext == "pkl":
    #     import pickle
    #     print(args.overwrite)
    #     if args.overwrite:
    #         mode = "wb"
    #     else:
    #         mode = "ab"

    #     with open(args.output, mode) as output:
    #         pickle.dump(s, output, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    cli()