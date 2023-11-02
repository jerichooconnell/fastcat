# Write a script that converts a reads an mhd file and makes a phantom

import numpy as np
import SimpleITK as sitk
import os
import tigre
from fastcat.simulate import data_path
from fastcat.spectrum import log_interp_1d
from fastcat import phantoms

import logging

def read_mhd(filename):
    """Reads an mhd file and returns a numpy array"""
    itkimage = sitk.ReadImage(filename)
    numpyImage = sitk.GetArrayFromImage(itkimage)
    numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))
    numpySpacing = np.array(list(reversed(itkimage.GetSpacing())))
    return numpyImage, numpyOrigin, numpySpacing

def read_range_file(filename):
    """Reads a range file and returns a list of ranges"""
    materials = []
    low_range = []
    high_range = []
    with open(filename) as f:
        for line in f:
            words = line.split(' ')
            if len(words) == 3:
                materials.append(words[2])
                low_range.append(float(words[0]))
                high_range.append(float(words[1]))

    return np.array(materials), np.array(low_range), np.array(high_range)

def get_phantom_from_mhd(filename, range_file, material_file=None):
    """Reads an mhd file and returns a phantom object"""
    numpyImage, numpyOrigin, numpySpacing = read_mhd(filename)
    phantom = phantoms.Phantom()
    phantom.phantom = numpyImage
    phantom.geomet = tigre.geometry_default(nVoxel=phantom.phantom.shape)
    phantom.geomet.DSD = 1510
    phantom.geomet.dVoxel = numpySpacing
    phantom.geomet.sVoxel = phantom.geomet.dVoxel*phantom.geomet.nVoxel
    phantom.geomet.dDetector = np.array([0.784,0.784])
    phantom.geomet.nDetector = np.array([512,512])
    phantom.geomet.sDetector = phantom.geomet.dDetector*phantom.geomet.nDetector
    phantom.is_non_integer = False

    if range_file is not None:
        materials, low_range, high_range = read_range_file(range_file)
        phantom.phan_map = materials
        if low_range != high_range:
            print('Warning: range file contains ranges. Using low range for all materials')
    
    if material_file is not None:
        make_material_mu_files(material_file)

    return phantom

# A function that fetches element atomic number from the element name
def name_to_atomic_number(name):
    element_base = get_element_base()
    # Search for a match
    for atomic_number, element in element_base.items():
        if name.lower().split(' ')[0] in element[0].lower():
            return atomic_number
    # If no match was found, return None
    return None
# %%
def get_element_base():
    element_base = {
        # number: name symbol common_ions uncommon_ions
        # ion info comes from Wikipedia: list of oxidation states of the elements.
        0: ['Neutron',     'n',  [],         []],
        1: ['Hydrogen',    'H',  [-1, 1],    []],
        2: ['Helium',      'He', [],         [1, 2]],  # +1,+2  http://periodic.lanl.gov/2.shtml
        3: ['Lithium',     'Li', [1],        []],
        4: ['Beryllium',   'Be', [2],        [1]],
        5: ['Boron',       'B',  [3],        [-5, -1, 1, 2]],
        6: ['Carbon',      'C',  [-4, -3, -2, -1, 1, 2, 3, 4], []],
        7: ['Nitrogen',    'N',  [-3, 3, 5], [-2, -1, 1, 2, 4]],
        8: ['Oxygen',      'O',  [-2],       [-1, 1, 2]],
        9: ['Fluorine',    'F',  [-1],       []],
        10: ['Neon',       'Ne', [],         []],
        11: ['Sodium',     'Na', [1],        [-1]],
        12: ['Magnesium',  'Mg', [2],        [1]],
        13: ['Aluminum',   'Al', [3],        [-2, -1, 1, 2]],
        14: ['Silicon',    'Si', [-4, 4],    [-3, -2, -1, 1, 2, 3]],
        15: ['Phosphorus', 'P',  [-3, 3, 5], [-2, -1, 1, 2, 4]],
        16: ['Sulfur',     'S',  [-2, 2, 4, 6],    [-1, 1, 3, 5]],
        17: ['Chlorine',   'Cl', [-1, 1, 3, 5, 7], [2, 4, 6]],
        18: ['Argon',      'Ar', [],         []],
        19: ['Potassium',  'K',  [1],        [-1]],
        20: ['Calcium',    'Ca', [2],        [1]],
        21: ['Scandium',   'Sc', [3],        [1, 2]],
        22: ['Titanium',   'Ti', [4],        [-2, -1, 1, 2, 3]],
        23: ['Vanadium',   'V',  [5],        [-3, -1, 1, 2, 3, 4]],
        24: ['Chromium',   'Cr', [3, 6],     [-4, -2, -1, 1, 2, 4, 5]],
        25: ['Manganese',  'Mn', [2, 4, 7],  [-3, -2, -1, 1, 3, 5, 6]],
        26: ['Iron',       'Fe', [2, 3, 6],  [-4, -2, -1, 1, 4, 5, 7]],
        27: ['Cobalt',     'Co', [2, 3],     [-3, -1, 1, 4, 5]],
        28: ['Nickel',     'Ni', [2],        [-2, -1, 1, 3, 4]],
        29: ['Copper',     'Cu', [2],        [-2, 1, 3, 4]],
        30: ['Zinc',       'Zn', [2],        [-2, 1]],
        31: ['Gallium',    'Ga', [3],        [-5, -4, -2, -1, 1, 2]],
        32: ['Germanium',  'Ge', [-4, 2, 4], [-3, -2, -1, 1, 3]],
        33: ['Arsenic',    'As', [-3, 3, 5], [-2, -1, 1, 2, 4]],
        34: ['Selenium',   'Se', [-2, 2, 4, 6], [-1, 1, 3, 5]],
        35: ['Bromine',    'Br', [-1, 1, 3, 5], [4, 7]],
        36: ['Krypton',    'Kr', [2],        []],
        37: ['Rubidium',   'Rb', [1],        [-1]],
        38: ['Strontium',  'Sr', [2],        [1]],
        39: ['Yttrium',    'Y',  [3],        [1, 2]],
        40: ['Zirconium',  'Zr', [4],        [-2, 1, 2, 3]],
        41: ['Niobium',    'Nb', [5],        [-3, -1, 1, 2, 3, 4]],
        42: ['Molybdenum', 'Mo', [4, 6],     [-4, -2, -1, 1, 2, 3, 5]],
        43: ['Technetium', 'Tc', [4, 7],     [-3, -1, 1, 2, 3, 5, 6]],
        44: ['Ruthenium',  'Ru', [3, 4],     [-4, -2, 1, 2, 5, 6, 7, 8]],
        45: ['Rhodium',    'Rh', [3],        [-3, -1, 1, 2, 4, 5, 6]],
        46: ['Palladium',  'Pd', [2, 4],     [1, 3, 5, 6]],
        47: ['Silver',     'Ag', [1],        [-2, -1, 2, 3, 4]],
        48: ['Cadmium',    'Cd', [2],        [-2, 1]],
        49: ['Indium',     'In', [3],        [-5, -2, -1, 1, 2]],
        50: ['Tin',        'Sn', [-4, 2, 4], [-3, -2, -1, 1, 3]],
        51: ['Antimony',   'Sb', [-3, 3, 5], [-2, -1, 1, 2, 4]],
        52: ['Tellurium',  'Te', [-2, 2, 4, 6], [-1, 1, 3, 5]],
        53: ['Iodine',     'I',  [-1, 1, 3, 5, 7], [4, 6]],
        54: ['Xenon',      'Xe', [2, 4, 6],  [8]],
        55: ['Caesium',     'Cs', [1],        [-1]],
        56: ['Barium',     'Ba', [2],        [1]],
        57: ['Lanthanum',  'La', [3],        [1, 2]],
        58: ['Cerium',     'Ce', [3, 4],     [2]],
        59: ['Praseodymium', 'Pr', [3],      [2, 4, 5]],
        60: ['Neodymium',  'Nd', [3],        [2, 4]],
        61: ['Promethium', 'Pm', [3],        [2]],
        62: ['Samarium',   'Sm', [3],        [2]],
        63: ['Europium',   'Eu', [2, 3],     []],
        64: ['Gadolinium', 'Gd', [3],        [1, 2]],
        65: ['Terbium',    'Tb', [3],        [1, 2, 4]],
        66: ['Dysprosium', 'Dy', [3],        [2, 4]],
        67: ['Holmium',    'Ho', [3],        [2]],
        68: ['Erbium',     'Er', [3],        [2]],
        69: ['Thulium',    'Tm', [3],        [2]],
        70: ['Ytterbium',  'Yb', [3],        [2]],
        71: ['Lutetium',   'Lu', [3],        [2]],
        72: ['Hafnium',    'Hf', [4],        [-2, 1, 2, 3]],
        73: ['Tantalum',   'Ta', [5],        [-3, -1, 1, 2, 3, 4]],
        74: ['Tungsten',   'W',  [4, 6],     [-4, -2, -1, 1, 2, 3, 5]],
        75: ['Rhenium',    'Re', [4],        [-3, -1, 1, 2, 3, 5, 6, 7]],
        76: ['Osmium',     'Os', [4],        [-4, -2, -1, 1, 2, 3, 5, 6, 7, 8]],
        77: ['Iridium',    'Ir', [3, 4],     [-3, -1, 1, 2, 5, 6, 7, 8, 9]],
        78: ['Platinum',   'Pt', [2, 4],     [-3, -2, -1, 1, 3, 5, 6]],
        79: ['Gold',       'Au', [3],        [-3, -2, -1, 1, 2, 5]],
        80: ['Mercury',    'Hg', [1, 2],     [-2, 4]],  # +4  doi:10.1002/anie.200703710
        81: ['Thallium',   'Tl', [1, 3],     [-5, -2, -1, 2]],
        82: ['Lead',       'Pb', [2, 4],     [-4, -2, -1, 1, 3]],
        83: ['Bismuth',    'Bi', [3],        [-3, -2, -1, 1, 2, 4, 5]],
        84: ['Polonium',   'Po', [-2, 2, 4], [5, 6]],
        85: ['Astatine',   'At', [-1, 1],    [3, 5, 7]],
        86: ['Radon',      'Rn', [2],        [6]],
        87: ['Francium',   'Fr', [1],        []],
        88: ['Radium',     'Ra', [2],        []],
        89: ['Actinium',   'Ac', [3],        []],
        90: ['Thorium',        'Th', [4],    [1, 2, 3]],
        91: ['Protactinium',   'Pa', [5],    [3, 4]],
        92: ['Uranium',        'U',  [6],    [1, 2, 3, 4, 5]],
        93: ['Neptunium',      'Np', [5],    [2, 3, 4, 6, 7]],
        94: ['Plutonium',      'Pu', [4],    [2, 3, 5, 6, 7]],
        95: ['Americium',      'Am', [3],    [2, 4, 5, 6, 7]],
        96: ['Curium',         'Cm', [3],    [4, 6]],
        97: ['Berkelium',      'Bk', [3],    [4]],
        98: ['Californium',    'Cf', [3],    [2, 4]],
        99: ['Einsteinium',    'Es', [3],    [2, 4]],
        100: ['Fermium',       'Fm', [3],    [2]],
        101: ['Mendelevium',   'Md', [3],    [2]],
        102: ['Nobelium',      'No', [2],    [3]],
        103: ['Lawrencium',    'Lr', [3],    []],
        104: ['Rutherfordium', 'Rf', [4],    []],
        105: ['Dubnium',       'Db', [5],    []],
        106: ['Seaborgium',    'Sg', [6],    []],
        107: ['Bohrium',       'Bh', [7],    []],
        108: ['Hassium',       'Hs', [8],    []],
        109: ['Meitnerium',    'Mt', [],     []],
        110: ['Darmstadtium',  'Ds', [],     []],
        111: ['Roentgenium',   'Rg', [],     []],
        112: ['Copernicium',   'Cn', [2],    []],
        113: ['Nihonium',      'Nh', [],     []],
        114: ['Flerovium',     'Fl', [],     []],
        115: ['Moscovium',     'Mc', [],     []],
        116: ['Livermorium',   'Lv', [],     []],
        117: ['Tennessine',    'Ts', [],     []],
        118: ['Oganesson',     'Og', [],     []],
    }
    return element_base

def get_element_density():
    # Get the path to the density data file
    density_file = os.path.join(data_path,'density.npy')

    # Check if a file containing the density data exists
    if not os.path.isfile(density_file):
        # If not, fetch the data from the internet
        print('Fetching density data from the internet')
        try:
            from physdata.xray import fetch_elements  # pip install physdata
        except:
            print('Could not import physdata. Please install it with "pip install physdata"')
            return
        element_density = []

        for element in fetch_elements():   
            element_density.append(element.density)

        # Save the data to a file
        np.save(density_file, element_density)
    else:
        # If it does, load the data from the file
        logging.info('Loading density data from file')
        element_density = np.load(density_file)
    
    return element_density

def make_material_mu_files(material_file):

    element_density = get_element_density()

    names = []
    weights = []
    elements = []
    densities = []

    with open(material_file,'r') as f:
        
        lines = f.readlines()
            
        for ii in range(len(lines)):
            
            line = lines[ii]
            
            if 'd=' in line:
                
                name = line.split(':')[0]
                rest = line.split(':')[1]

                density = float(rest.split(';')[0].split('=')[1].split(' ')[0])

                jj = ii + 1
                
                weight = []
                element = []
                
                while jj < len(lines) and '+el' in lines[jj]:
                    
                    element.append(lines[jj].split('=')[1].split(';')[0])
                    weight.append(float(lines[jj].split('=')[-1]))
                    
                    jj += 1
                
                elements.append(element)
                weights.append(weight)   
                names.append(name)
                densities.append(density)

    H = np.loadtxt(os.path.join(data_path,'mu','1.csv'),delimiter=',')

    for kk, name in enumerate(names):
        
        attenuations = []
        energies = []
        lens = []
                            
        for weight, element in zip(weights[kk],elements[kk]):
            
            atomic_number = name_to_atomic_number(element)

            attenuation = np.loadtxt(os.path.join(data_path,'mu',str(atomic_number)+'.csv'),delimiter=',')[1]
            attenuation = attenuation/element_density[atomic_number-1]*weight
            
            attenuations.append(attenuation)
            energies.append(np.loadtxt(os.path.join(data_path,'mu',str(atomic_number)+'.csv'),delimiter=',')[0])
            lens.append(len(attenuation))
            
        if max(lens) == 36:
            
            attenuation_all = np.zeros([2,len(H[0])])
            attenuation_all[0] = H[0]
            
            for ii in range(len(attenuations)):
                
                attenuation_all[1] += attenuations[ii]
            
        else:
            
            energies_larger = np.unique(np.hstack(energies))
            
            attenuation_all = np.zeros([2,len(energies_larger)])
            attenuation_all[0] = energies_larger
            
            for ii in range(len(attenuations)):
                
                f = log_interp_1d(energies[ii],attenuations[ii])
                
                attenuation_all[1] += f(energies_larger)
                
            
        attenuation_all[1] *= densities[kk]
        np.savetxt(os.path.join(data_path,'mu',name+".csv"),attenuation_all, fmt="%.8G",delimiter=",")
        logging.info(f'    Saved {name} atten to file in data/mu/{name}.csv')
