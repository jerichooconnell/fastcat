import numpy as np
from numpy import cos, sin
from astropy import stats
import matplotlib.pyplot as plt

star_data_gos = [0, -745.8193979933105,
0, -709.0301003344478,
0.854368932038835, -30.100334448160538,
0.9449838187702266, 43.478260869565474,
0.9967637540453074, 113.71237458193991,
1.1477885652642934, 197.32441471571929,
1.3592233009708738, 394.64882943143834,
1.866235167206041, 842.8093645484951
]
star_data_cwo = [0.0041972717733472775, -864.2384105960264,
0.0041972717733472775, -880.7947019867552,
0.8541448058761805, -105.96026490066242,
0.9443861490031478, -3.3112582781459423,
1.0031479538300103, 39.73509933774835,
1.1521511017838406, 152.31788079470198,
1.359916054564533, 364.23841059602637,
1.869884575026233, 837.7483443708609]

def create_mask(shape, off = [0,0], r = 6, radius = 0.4, rot = 0):

    im = np.zeros([512,512])
    #im = np.zeros([256,256])
    ii = 1

    # CTMAT(x) formel=H2O dichte=x
    LEN = 100

    A0  = (90.0+ rot)*np.pi/180

    # Phantom 
    # ++++ module body ++++++++++++++++++++++++++++++++++++++++++++++++++ */
    
    create_circular_mask(x= 6.5*cos(A0+1/6*np.pi),  y= 6.5*sin(A0+1/6*np.pi),  r=1.0, index = ii, image = im)

    rad = r 

    for ii,jj in enumerate([2,4,6,8,10,12]):

        # ++++ supra-slice 1.0% targets +++++++++++++++++++++++++++++++++++++++ */
        create_circular_mask(x= rad*cos(A0+jj/6*np.pi) - off[0],  y= rad*sin(A0+jj/6*np.pi) - off[1],  r=radius, index = ii +2, image = im)

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

def return_CNR(recon_slice,im,show_map=False):
    '''
    Returns: contrast, CNR, noise
    '''
    contrast = []
    CNR = []
    noise = []

    for ii in range(2,int(np.max(im)+1)):

        contrast.append(np.mean(recon_slice[im == ii]))
        CNR.append(contrast[-1]/np.sqrt(np.std(recon_slice[im == ii])**2 + np.std(recon_slice[im == 6])**2))
        noise.append(np.std(recon_slice[im == ii]))
        
    if show_map:
        plt.figure()
        plt.imshow(np.where(im==0,recon_slice,np.zeros_like(recon_slice)))

    return np.array(sorted(contrast)), np.array(CNR)[np.argsort(contrast)], np.array(noise)[np.argsort(contrast)]