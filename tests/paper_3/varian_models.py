import numpy as np
from analysis_functions import *
import fastcat.fastcat as fc
from scipy.optimize import minimize

class Catphan_404_kV(fc.Catphan_404):
    '''
    Making another function that returns the exact kV data
    '''
    def __init__(self):
        
        fc.Catphan_404.__init__(self) # inheret Catphan parameters
        s = fc.Spectrum()
        s.x = np.load('data/energies.npy')
        s.y = np.load('data/spec_100.npy') # don't calculate too slow
        s.attenuate(0.4,fc.get_mu(z=13)) # 4 mm al
        s.attenuate(0.089,fc.get_mu(z=22)) # 0.89 mm Ti Varian OBI beam hardening
        self.s = s
        # Define phantom map the two base materials are ad libbed since
        # Phantom Lab doesn't give compositions e density in Star-Lack paper
        self.phan_map = ['air','G4_POLYSTYRENE','G4_POLYVINYL_BUTYRAL','G4_POLYVINYL_BUTYRAL','CATPHAN_Delrin','G4_POLYVINYL_BUTYRAL','CATPHAN_Teflon_revised','air','CATPHAN_PMP','G4_POLYVINYL_BUTYRAL','CATPHAN_LDPE','G4_POLYVINYL_BUTYRAL','CATPHAN_Polystyrene','air','CATPHAN_Acrylic','air','CATPHAN_Teflon','air','air','air','air'] 
        self.geomet.DSD = 1520 # This is what it is for some reason
        self.kernel = fc.Kernel(s,'CsI-784-micrometer')
        self.kernel.add_focal_spot(1.2) # Should be 1.0 actually
#         self.geomet.nVoxel = np.array([10,512,512])
        self.geomet.dVoxel = self.geomet.sVoxel/self.geomet.nVoxel
        
    def get_proj(self,angles,fudge=0.3,ASG=False,return_dose = False,nphoton=None,bowtie=True):
        
        self.angles = angles
        self.res = self.return_projs(self.kernel,self.s,
                          angles,mgy = fudge*21.9/917,
                          scat_on=True,bowtie=bowtie,
                          filter='bowtie3',ASG=ASG,
                          return_dose = return_dose,
                                    nphoton=nphoton)
        
#         if recon:
#             self.reconstruct('FDK',filt='ram_lak')
            
    def plot_one_proj(self,ref,images=False):
        
        fastcat_im = np.roll((self.proj[0]),-3)*1.03 #

        dist = np.linspace(-256*0.0784 - 0.0392,256*0.0784 - 0.0392, 512)
        dist2 = np.linspace(-256*0.0776- 0.0388,256*0.0776 - 0.0388, 512)
        bcca_smaller_im = ref
        if images:
            plt.figure()
            plt.subplot(121)
            plt.imshow(fastcat_im.T,cmap = 'gray')
            plt.subplot(122)
            plt.imshow(10*bcca_smaller_im.T,cmap='gray')
            plt.tight_layout()

        plt.figure()
        plt.plot(dist2,(10*bcca_smaller_im[5]),c='darkorange',lw=0.7)
        plt.plot(dist,(fastcat_im[5]),c='cornflowerblue',lw=0.7)
        
    def return_CNR(self,return_contrast = False):
        
        im = create_mask(self.img[64].shape,r=5.75,radius=0.25)
        contrast_fc, CNR_fc, noise_fc = return_CNR(self.img[64],im)
#         print(contrast_fc ,'fastcat')
        if return_contrast:
            return contrast_fc
        else:
            return CNR_fc
        
    def return_CNR_exp(self,return_contrast = False):
        
        recon_slice = np.rot90(self.img[64],-1)
        im = create_mask(recon_slice.shape,radius=0.25,r=5.85,off = [-0.15,-0.02],rot = 0.7)
        contrast_fc, CNR_fc, noise_fc = return_CNR(recon_slice,im)
#         print(contrast_fc, 'exp')
        if return_contrast:
            return contrast_fc
        else:
            return CNR_fc

class Catphan_404_6x(fc.Catphan_404):
    '''
    Making another function that returns the exact kV data
    '''
    def __init__(self):
        
        fc.Catphan_404.__init__(self) # inheret Catphan parameters
        s = fc.Spectrum()
        s.load('Varian_truebeam')
        s.x[0] = 1; s.x[1] = 2
#         s.attenuate(0.4,fc.get_mu(z=74)) # 4 mm al
#         s.load('Varian_truebeam_phasespace')
        self.s = s
        # Define phantom map the two base materials are ad libbed since
        # Phantom Lab doesn't give compositions e density in Star-Lack paper
        self.phan_map = ['air','G4_POLYSTYRENE','G4_POLYVINYL_BUTYRAL','G4_POLYVINYL_BUTYRAL','CATPHAN_Delrin','G4_POLYVINYL_BUTYRAL','CATPHAN_Teflon_revised','air','CATPHAN_PMP','G4_POLYVINYL_BUTYRAL','CATPHAN_LDPE','G4_POLYVINYL_BUTYRAL','CATPHAN_Polystyrene','air','CATPHAN_Acrylic','air','CATPHAN_Teflon','air','air','air','air'] 
        self.geomet.DSD = 1520 # This is what it is for some reason
        self.kernel = fc.Kernel(s,'CuGOS-784-micrometer')
        self.kernel.add_focal_spot(1.2) # Should be 1.0 actually
#         self.geomet.nVoxel = np.array([10,512,512])
        self.geomet.dVoxel = self.geomet.sVoxel/self.geomet.nVoxel
        
    def get_proj(self,angles,fudge=0.3,return_dose = False, nphoton=None,bowtie=True):
        
        self.angles = angles
        self.res = self.return_projs(self.kernel,self.s,
                          angles,mgy = fudge*3000/(505),
                          scat_on=True,bowtie=bowtie,filter='FF0',
                                    return_dose = return_dose,
                                    nphoton = nphoton)
            
    def plot_one_proj(self,ref,images=False):
        
        fastcat_im = np.roll((self.proj[0]),-3)*1.03 #

        dist = np.linspace(-256*0.0784 - 0.0392,256*0.0784 - 0.0392, 512)
        dist2 = np.linspace(-256*0.0776- 0.0388,256*0.0776 - 0.0388, 512)
        bcca_smaller_im = ref
        if images:
            plt.figure()
            plt.subplot(121)
            plt.imshow(fastcat_im.T,cmap = 'gray')
            plt.subplot(122)
            plt.imshow(10*bcca_smaller_im.T,cmap='gray')
            plt.tight_layout()

        plt.figure()
        plt.plot(dist2,(10*bcca_smaller_im[5]),c='darkorange',lw=0.7)
        plt.plot(dist,(fastcat_im[5]),c='cornflowerblue',lw=0.7)
        
    def return_CNR(self,return_contrast = False):
        
        im = create_mask(self.img[64].shape,r=5.75,radius=0.25)
        contrast_fc, CNR_fc, noise_fc = return_CNR(self.img[64],im)
#         print(contrast_fc ,'fastcat')
        if return_contrast:
            return contrast_fc
        else:
            return CNR_fc
        
    def return_CNR_exp(self,return_contrast = False):
        
        recon_slice = np.rot90(self.img[64],-1)
        im = create_mask(recon_slice.shape,radius=0.25,r=5.75,off = [-0.15,-0.02])
        contrast_fc, CNR_fc, noise_fc = return_CNR(recon_slice,im)
#         print(contrast_fc, 'exp')
        if return_contrast:
            return contrast_fc
        else:
            return CNR_fc       

def return_CNR_fc3(img,return_contrast = False):

    im = create_mask(img.shape,r=5.75,radius=0.25)
    contrast_fc, CNR_fc, noise_fc = return_CNR(img,im)#,show_map=True)
#         print(contrast_fc ,'fastcat')
    if return_contrast:
        return contrast_fc
    else:
        return CNR_fc

def return_CNR_exp3(img,return_contrast = False):

    recon_slice = np.rot90(img,-2)
    im = create_mask(recon_slice.shape,radius=0.25,r=5.75,off = [-0.15,-0.0])
    contrast_fc, CNR_fc, noise_fc = return_CNR(recon_slice,im)#,True)
#         print(contrast_fc, 'exp')
    if return_contrast:
        return contrast_fc
    else:
        return CNR_fc
        
def return_CNR_fc2(img,return_contrast = False):

    im = create_mask(img.shape,r=5.75,radius=0.25)
    contrast_fc, CNR_fc, noise_fc = return_CNR(img,im)
#         print(contrast_fc ,'fastcat')
    if return_contrast:
        return contrast_fc
    else:
        return CNR_fc

def return_CNR_exp2(img,return_contrast = False):

    recon_slice = np.rot90(img,-1)
    im = create_mask(recon_slice.shape,radius=0.25,r=5.85,off = [-0.15,-0.02],rot = 0.7)
    contrast_fc, CNR_fc, noise_fc = return_CNR(recon_slice,im)
#         print(contrast_fc, 'exp')
    if return_contrast:
        return contrast_fc
    else:
        return CNR_fc
        