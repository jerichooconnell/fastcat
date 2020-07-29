import os, shutil
import itertools
import imageio

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from scipy.io import loadmat
from scipy.ndimage import gaussian_filter1d
from scipy import interpolate
from scipy import linalg

from sklearn.covariance import EmpiricalCovariance, MinCovDet
from sklearn.decomposition import NMF, FastICA, PCA
from sklearn.metrics import homogeneity_score,homogeneity_completeness_v_measure
from sklearn import mixture
from skimage.restoration import inpaint
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans


import scipy.ndimage as ndimage
import scipy.spatial as spatial
import scipy.misc as misc
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, Ellipse

class I_gmm:
    def __init__(self):
        self.XY = None
        
    def plot_cov(self,means, covariances,ct):
        if ct == 'spherical':
            return
        color_iter = itertools.cycle(['navy', 'navy', 'cornflowerblue', 'gold',
                                  'darkorange'])
        ax =plt.gca()
        for i, (mean, covar, color) in enumerate(zip(
                means, covariances, color_iter)):
                
            v, w = linalg.eigh(covar)
            v = 2. * np.sqrt(2.) * np.sqrt(v)
            u = w[0] / linalg.norm(w[0])

            # Plot an ellipse to show the Gaussian component
            angle = np.arctan(u[1] / u[0])
            angle = 180. * angle / np.pi  # convert to degrees
            alpha = 0.2
            ell = Ellipse(mean, v[0], v[1], 180. + angle, color=color)
            ell.set_clip_box(ax.bbox)
            ell.set_alpha(alpha)
            ax.add_artist(ell)
            ell = Ellipse(mean, v[0]*4, v[1]*2, 180. + angle, color=color)
            ell.set_clip_box(ax.bbox)
            ell.set_alpha(alpha)
            ell = Ellipse(mean, v[0]*2, v[1]*2, 180. + angle, color=color)
            ell.set_clip_box(ax.bbox)
            ell.set_alpha(alpha)
            ax.add_artist(ell)            
            ax.scatter(mean[0],mean[1],c='grey',zorder=10,s=100)        
        
    def iterative_gmm(self,X1,fake = True,mode2 = 'gmm',mode=[],binary = False,im_dir = './images/',savegif = False,title ='temp',bic_thresh = 0,maxiter = 40,nc =5,v_and_1 = False,thresh = 0.9,cov=[],n_components=2,covt='spherical',ra=False,red = 'pca'):

        '''
        dataset: string
        The filename of the material something like 'bb','pp'
        fake: bool
        Whether or not the data is fake, if it is not it will be cropped
        mode: str
        'fraction' will reduce the input to a combination of the relative signals
        e.g. bin1 - bin0/sum
        binary: bool
        Whether or not to show the output as binary or not
        nc: int
        pca components
        '''
            
        gmm = mixture.GaussianMixture(n_components=n_components,
                                      covariance_type=covt)
        gmm1 = mixture.GaussianMixture(n_components=n_components,
                                      covariance_type='full')
        
        bic0 = 0
     
        for ii in range(0,maxiter):

            X = X1.copy()

            y_pred = gmm.fit_predict(X)

            
            y_ff1 = gmm1.fit(X)
            

            bic = gmm.aic(X)


            # Stop if bic is lower
            if bic - bic0 < bic_thresh:
                bic0 = bic
            else:
                print('BIC got higher')
                break
            
            # map the bad values to zero
            for kk in range(n_components):
                
                temp = X[y_pred == kk,:]

                robust_cov = EmpiricalCovariance().fit(temp)
                
                # Calculating the mahal distances
                robust_mahal = robust_cov.mahalanobis(
                    temp - robust_cov.location_) ** (0.33)
                
                if thresh < 1:
                    temp[robust_mahal > robust_mahal.min() + (robust_mahal.max()-robust_mahal.min()) *thresh] = 0
                else:
                    temp[robust_mahal > np.sort(robust_mahal)[-thresh]] = 0
                    
                X[y_pred == kk,:] = temp

            mask_one = X[:,0] == 0

            m_reshape = np.reshape(mask_one, (20,length), order="F")

            X2 = X1.copy()
            # Inpainting the zeros
            r2 = np.reshape(X1, (20,length,X1.shape[1]), order="F")
            X1 = np.reshape(inpaint.inpaint_biharmonic(
                r2,m_reshape,multichannel=True),
                            (20*length,X1.shape[1]),order="F")

        return X1

    def find_paws(self,data, smooth_radius = 1, threshold = 0.0001):
        # https://stackoverflow.com/questions/4087919/how-can-i-improve-my-paw-detection
        """Detects and isolates contiguous regions in the input array"""
        # Blur the input data a bit so the paws have a continous footprint 
        data = ndimage.uniform_filter(data, smooth_radius)
        # Threshold the blurred data (this needs to be a bit > 0 due to the blur)
        thresh = data > threshold
        # Fill any interior holes in the paws to get cleaner regions...
        filled = ndimage.morphology.binary_fill_holes(thresh)
        # Label each contiguous paw
        coded_paws, num_paws = ndimage.label(filled)
        # Isolate the extent of each paw
        # find_objects returns a list of 2-tuples: (slice(...), slice(...))
        # which represents a rectangular box around the object
        data_slices = ndimage.find_objects(coded_paws)
        return data_slices

    def animate(self,frame,im = None):
        """Detects paws and animates the position and raw data of each frame
        in the input file"""
        # With matplotlib, it's much, much faster to just update the properties
        # of a display object than it is to create a new one, so we'll just update
        # the data and position of the same objects throughout this animation...

        # Since we're making an animation with matplotlib, we need 
        # ion() instead of show()...
        fig = plt.gcf()
        ax = plt.axes([.25, .55, .6, .4], facecolor='y')
        plt.axis('off')

        # Make an image based on the first frame that we'll update later
        # (The first frame is never actually displayed)
        if im is None:
            plt.imshow(frame,cmap='brg')
        else:
            plt.imshow(im)
        plt.title('Image Space')

        # Make 4 rectangles that we can later move to the position of each paw
        rects = [Rectangle((0,0), 1,1, fc='none', ec='red') for i in range(4)]
        [ax.add_patch(rect) for rect in rects]


        # Process and display each frame

        paw_slices = self.find_paws(frame)

        # Hide any rectangles that might be visible
        [rect.set_visible(False) for rect in rects]

        # Set the position and size of a rectangle for each paw and display it
        for slice, rect in zip(paw_slices, rects):
            dy, dx = slice
            rect.set_xy((dx.start, dy.start))
            rect.set_width(dx.stop - dx.start + 1)
            rect.set_height(dy.stop - dy.start + 1)
            rect.set_visible(True)