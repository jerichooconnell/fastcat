# Write a script that converts a reads an mhd file and makes a phantom

import numpy as np
import SimpleITK as sitk
import os
import tigre
from fastcat.simulate import data_path
from fastcat.spectrum import Spectrum, log_interp_1d
from fastcat.ggems_simulate import Phantom

import logging
import nrrd


def read_mhd(filename):
    """Reads an mhd file and returns a numpy array"""
    itkimage = sitk.ReadImage(filename)
    numpyImage = sitk.GetArrayFromImage(itkimage)
    numpyImage = np.flip(np.flipud(numpyImage), axis=2)
    numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))
    numpySpacing = np.array(list(reversed(itkimage.GetSpacing())))
    return numpyImage, numpyOrigin, numpySpacing


def read_mhd2(filename):
    """Reads an mhd file and returns a numpy array"""
    itkimage = sitk.ReadImage(filename)
    numpyImage = sitk.GetArrayFromImage(itkimage)
    numpyImage = np.flip(np.flipud(numpyImage), axis=2)
    return numpyImage.squeeze()


def read_range_file(filename):
    """Reads a range file and returns a list of ranges"""
    materials = []
    low_range = []
    high_range = []
    with open(filename) as f:
        for line in f:
            words = line.split()
            if len(words) == 3:
                # Check if the material is caesium
                if words[2].lower() == 'caesium':
                    words[2] = 'cesium'
                materials.append(words[2])
                low_range.append(float(words[0]))
                high_range.append(float(words[1]))

    return np.array(materials), np.array(low_range), np.array(high_range)


def write_range_file(filename, materials):
    """Writes a range file"""

    range_file = os.path.join(
        data_path, 'user_phantoms', filename.split(".")[0], f'{filename.split(".")[0]}_range.txt')
    
    # Check if the directory exists
    if not os.path.exists(os.path.dirname(range_file)):
        os.makedirs(os.path.dirname(range_file))
    
    with open(range_file, 'wt') as f:
        for i in range(len(materials)):
            f.write(f'{i} {i} {materials[i]}\n')

def write_mhd_file(filename, numpyImage, numpyOrigin, numpySpacing):
    """Writes a numpy array to an mhd file"""
    itkimage = sitk.GetImageFromArray(numpyImage)
    itkimage.SetOrigin(numpyOrigin[::-1])
    itkimage.SetSpacing(numpySpacing[::-1])
    mhd_file = os.path.join(
        data_path, 'user_phantoms', filename.split(".")[0], f'{filename.split(".")[0]}_phantom.mhd')
    sitk.WriteImage(itkimage, mhd_file)

def get_phantom_from_mhd(filename, range_file, material_file=None,geo=None):
    """Reads an mhd file and returns a phantom object"""
    numpyImage, numpyOrigin, numpySpacing = read_mhd(filename)
    phantom = Phantom()
    phantom.mhd_file = filename
    phantom.range_file = range_file
    phantom.material_file = material_file
    phantom.phantom = numpyImage
    phantom.geomet = tigre.geometry_default(nVoxel=phantom.phantom.shape)

    if geo == None:
        logging.info('Using default geometry, DSD = 1510, nDetector = [512, 512], dDetector = [0.784, 0.784]')
        phantom.geomet.DSD = 1510
        phantom.geomet.dDetector = np.array([0.784, 0.784])
        phantom.geomet.nDetector = np.array([512, 512])
    else:
        logging.info(f'Using custom geometry, DSD = {geo.DSD}, nDetector = {geo.nDetector}, dDetector = {geo.dDetector}')
        phantom.geomet = geo
    
    phantom.geomet.sDetector = phantom.geomet.dDetector*phantom.geomet.nDetector
    phantom.is_non_integer = False
    phantom.geomet.dVoxel = numpySpacing
    phantom.geomet.sVoxel = phantom.geomet.dVoxel*phantom.geomet.nVoxel
    if range_file is not None:
        materials, low_range, high_range = read_range_file(range_file)
        phantom.phan_map = materials
        # check if all ranges are the same
        if np.any(low_range != high_range):
            print(
                'Warning: range file contains ranges. Using low range for all materials')

    if material_file is not None:
        make_material_mu_files(material_file)

    return phantom

# A function that fetches element atomic number from the element name
def get_phan_map_from_range(range_file):
    with open(range_file, 'r') as f:
        lines = f.readlines()

    materials = []
    for line in lines:
        parts = line.strip().split(' ')
        material_name = parts[2]
        materials.append(material_name)

    return materials

def name_to_atomic_number(name):
    element_base = get_element_base()
    # Search for a match
    for atomic_number, element in element_base.items():
        # Check if the element is 'Cesium' or 'Caesium'
        temp_name = name.lower().split(' ')[0]
        if temp_name == 'cesium':
            temp_name = 'caesium'
        if temp_name == 'phosphorus':
            temp_name = 'phosphor'

        if temp_name in element[0].lower():
            return atomic_number
    # If no match was found, return None
    return None


def init_ggems_scatter_simulation(simulation_directory=None, phantom_name=None, **kwargs):
    '''
    Small function for returning the scatter intensity
    profile from ggems simulation

    Can only be run after a simulation has been performed
    otherwise the variables won't be recognized
    '''

    if simulation_directory is None:
        range_file = kwargs['range_file']
        mhd_file = kwargs['mhd_file']
        material_file = kwargs['material_file']
        spectrum_file = kwargs['spectrum_file']
    else:
        range_file = os.path.join(
            simulation_directory, phantom_name, f'{phantom_name}_range.txt')
        mhd_file = os.path.join(simulation_directory,
                                phantom_name, f'{phantom_name}_phantom.mhd')
        material_file = os.path.join(
            simulation_directory, phantom_name, f'{phantom_name}_materials.txt')
        spectrum_file = os.path.join(
            simulation_directory, phantom_name, f'{phantom_name}_spectrum.dat')

    if simulation_directory is None:
        simulation_directory = os.path.dirname('.')
        phantom_name = mhd_file.split('/')[-1].split('.')[0]

    out_dir = os.path.join(simulation_directory, phantom_name, 'out')
    os.makedirs(out_dir, exist_ok=True)

    phantom = get_phantom_from_mhd(mhd_file, range_file, material_file)
    phantom.out_dir = out_dir
    phantom.is_ggems = True

    if os.path.exists(spectrum_file):
        s = Spectrum()
        s.load(spectrum_file=spectrum_file)
        return phantom, s
    else:
        return phantom, None
# %%


def get_element_base():
    element_base = {
        # number: name symbol common_ions uncommon_ions
        # ion info comes from Wikipedia: list of oxidation states of the elements.
        0: ['Neutron',     'n',  [],         []],
        1: ['Hydrogen',    'H',  [-1, 1],    []],
        # +1,+2  http://periodic.lanl.gov/2.shtml
        2: ['Helium',      'He', [],         [1, 2]],
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
        13: ['Aluminium',   'Al', [3],        [-2, -1, 1, 2]],
        14: ['Silicon',    'Si', [-4, 4],    [-3, -2, -1, 1, 2, 3]],
        15: ['Phosphor', 'P',  [-3, 3, 5], [-2, -1, 1, 2, 4]],
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
        # +4  doi:10.1002/anie.200703710
        80: ['Mercury',    'Hg', [1, 2],     [-2, 4]],
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
    density_file = os.path.join(data_path, 'density.npy')

    # Check if a file containing the density data exists
    if not os.path.isfile(density_file):
        # If not, fetch the data from the internet
        print('Fetching density data from the internet')
        try:
            from physdata.xray import fetch_elements  # pip install physdata
        except:
            print(
                'Could not import physdata. Please install it with "pip install physdata"')
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

    with open(material_file, 'r') as f:

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

    H = np.loadtxt(os.path.join(data_path, 'mu', '1.csv'), delimiter=',')

    for kk, name in enumerate(names):
        
        # Check if file already exists
        if os.path.exists(os.path.join(data_path, 'mu', name+".csv")):
            logging.info(f'    {name} atten file already exists, skipping...')
            continue

        attenuations = []
        energies = []
        lens = []

        for weight, element in zip(weights[kk], elements[kk]):

            atomic_number = name_to_atomic_number(element)

            attenuation = np.loadtxt(os.path.join(
                data_path, 'mu', str(atomic_number)+'.csv'), delimiter=',')[1]
            attenuation = attenuation/element_density[atomic_number-1]*weight

            attenuations.append(attenuation)
            energies.append(np.loadtxt(os.path.join(
                data_path, 'mu', str(atomic_number)+'.csv'), delimiter=',')[0])
            lens.append(len(attenuation))

        if max(lens) == 36:

            attenuation_all = np.zeros([2, len(H[0])])
            attenuation_all[0] = H[0]

            for ii in range(len(attenuations)):

                attenuation_all[1] += attenuations[ii]

        else:

            energies_larger = np.unique(np.hstack(energies))

            attenuation_all = np.zeros([2, len(energies_larger)])
            attenuation_all[0] = energies_larger

            for ii in range(len(attenuations)):

                f = log_interp_1d(energies[ii], attenuations[ii])

                attenuation_all[1] += f(energies_larger)

        attenuation_all[1] *= densities[kk]
        np.savetxt(os.path.join(data_path, 'mu', name+".csv"),
                   attenuation_all, fmt="%.8G", delimiter=",")
        logging.info(f'    Saved {name} atten to file in data/mu/{name}.csv')

def nrrd_to_mhd(nrrd_file, conversion_file='schneider_material_conv.txt',return_arrays=False,**kwargs):

    # Check if the nrrd file exists
    if not os.path.isfile(nrrd_file):
        logging.info('NRRD file not found')
        return
    
    # Check if the conversion file exists
    if not os.path.isfile(os.path.join(data_path, 'user_phantoms', conversion_file)):
        logging.info('Conversion file not found')
        return
    
    # Check if the directory exists
    if os.path.exists(os.path.join(data_path, 'user_phantoms', nrrd_file.split('/')[-1].split(".")[0])):
        if 'force' in kwargs:
            if kwargs['force']:
                logging.info(f'Directory {nrrd_file.split("/")[-1].split(".")[0]} already exists, but is being overwriten')
            else:
                logging.info(f'User phantom already exists, delete directory {nrrd_file.split("/")[-1].split(".")[0]} to overwrite')
                return
    
    conv_file = os.path.join(data_path, 'user_phantoms', conversion_file)
    # Read the text file
    with open(conv_file, 'r') as f:
        lines = f.readlines()
    # remove the newline character
    lines = [line.strip() for line in lines]
    # remove empty lines
    lines = [line for line in lines if line]
    # remove comments
    lines = [line for line in lines if not line.startswith('#')]
    # Parse the lines into variables
    HU_sections = list(map(int, lines[0].split('=')[1].split()[1:]))
    density_offset = list(map(float, lines[1].split('=')[1].split()[1:]))
    density_factor = list(map(float, lines[2].split('=')[1].split()[1:]))
    density_factor_offset = list(map(float, lines[3].split('=')[1].split()[1:]))
    elements = [el.split('"')[1] for el in lines[4].split('=')[1].split()[1:]]
    HU_to_material_sections = list(map(int, lines[5].split('=')[1].split()[1:]))
    materials_weight = [list(map(float, line.split('=')[1].split()[1:])) for line in lines[6:-1]]
    density_correction = list(map(float, lines[-1].split('=')[1].split()[1:-1]))
    names = [f"SchneiderMaterialsWeight{i+1}" for i in range(len(materials_weight))]

    # Define the function to decompose a DICOM array into a binary array of 25 materials
    def decompose_dicom(dicom_array):
        # Calculate the density array
        arr_clipped = np.clip(dicom_array, -1000, 2995)
        HU_vals = np.arange(-1000, 2996)
        density_map = np.zeros_like(density_correction)
        for ii, HU_section in enumerate(HU_sections[:-1] ):
            # print(ii,HU_sections[ii])
            # Formula: Density = (Offset + (Factor*(FactorOffset + HU[-1000,2995] ))) * DensityCorrection
            density_map[HU_sections[ii] + 1000:HU_sections[ii+1]+ 1000] = (
                density_offset[ii] + 
                (density_factor[ii] * (density_factor_offset[ii] +
                                    HU_vals[HU_sections[ii] + 1000:
                                            HU_sections[ii+1]+ 1000]))) * density_correction[HU_sections[ii] + 1000:
                                            HU_sections[ii+1]+ 1000]

        # Initialize the binary array
        binary_array = np.zeros_like(arr_clipped, dtype=int)

        # # Decompose the DICOM array into the binary array
        for i in range(25):
            inds = (HU_to_material_sections[i] <= arr_clipped) & (arr_clipped < HU_to_material_sections[i+1])
            binary_array[inds] = i

        # Map the HU values to density using the density map
        arr_clipped = arr_clipped + 1000

        # Map the HU values to the density values using the density map as a lookup table
        density_array = density_map[arr_clipped]

        return density_map, binary_array, density_array
       
    # Read the nrrd file
    nrrd_data, nrrd_header = nrrd.read(nrrd_file)
    numpyOrigin = nrrd_header['space origin']
    numpySpacing = nrrd_header['space directions']
    # Decompose the DICOM array
    density_map, binary_array, density_array = decompose_dicom(nrrd_data)
    average_densities = [np.mean(density_map[HU_to_material_sections[i]+1000:HU_to_material_sections[i+1]+1000]) for i in range(len(HU_to_material_sections[:-1]))]

    phantom_name = nrrd_file.split('/')[-1].split('.')[0]

    make_material_mu_files_schneider_all(names, elements, materials_weight, phantom_name, average_densities)
    write_range_file(phantom_name,names)
    write_mhd_file(phantom_name, binary_array.astype(np.uint32), numpyOrigin, np.abs(numpySpacing).max(axis=0))
    write_density_file(phantom_name, density_array)

    if return_arrays:
        return density_map, binary_array, density_array

def write_density_file(phantom_name, density_array):
    density_file = os.path.join(
        data_path, 'user_phantoms', phantom_name.split(".")[0], f'{phantom_name.split(".")[0]}_density.npy')
    np.save(density_file, density_array)

def write_material_file(material_file, names, densities, elements, weights):

    template_file = os.path.join(data_path, 'mu', 'template.txt')

    mat_file = os.path.join(data_path, 'user_phantoms', material_file.split(".")[0], f'{material_file.split(".")[0]}_materials.txt')
    
    # Check if the directory exists
    if not os.path.exists(os.path.dirname(mat_file)):
        os.makedirs(os.path.dirname(mat_file))
    
    with open(mat_file, 'w') as f:
        # Write the contents of the template file to the new material file
        with open(template_file, 'r') as f_template:
            f.write(f_template.read())

        f.write('\n')

        for ii in range(len(names)):
            
            print(
                f'{names[ii]}: d={densities[ii]:.4f} g/cm3 ; n={len(elements)};')
            f.write(
                f'{names[ii]}: d={densities[ii]:.4f} g/cm3 ; n={len(elements)};\n')

            for jj in range(len(elements)):
                if elements[jj] == 'Phosphorus':
                    elements[jj] = 'Phosphor'

                f.write(
                    f'\t+el={elements[jj]}; f={weights[ii][jj]:.4f}\n')

            f.write('\n')


def make_material_mu_files_schneider_all(names, elements, weights, material_file=None, densities=None):

    for name, weight in zip(names, weights):
        make_material_mu_files_schneider(name, elements, weight)
    if material_file is not None:
        write_material_file(material_file, names, densities, elements, weights)


def make_material_mu_files_schneider(name, elements, weights):

    element_density = get_element_density()

    H = np.loadtxt(os.path.join(data_path, 'mu', '1.csv'), delimiter=',')

    attenuations = []
    energies = []
    lens = []

    for weight, element in zip(weights, elements):
        atomic_number = name_to_atomic_number(element)
        attenuation = np.loadtxt(os.path.join(
            data_path, 'mu', str(atomic_number)+'.csv'), delimiter=',')[1]
        attenuation = attenuation/element_density[atomic_number-1]*weight

        attenuations.append(attenuation)
        energies.append(np.loadtxt(os.path.join(
            data_path, 'mu', str(atomic_number)+'.csv'), delimiter=',')[0])
        lens.append(len(attenuation))

    if max(lens) == 36:

        attenuation_all = np.zeros([2, len(H[0])])
        attenuation_all[0] = H[0]

        for ii in range(len(attenuations)):
            attenuation_all[1] += attenuations[ii]

    else:

        energies_larger = np.unique(np.hstack(energies))
        attenuation_all = np.zeros([2, len(energies_larger)])
        attenuation_all[0] = energies_larger

        for ii in range(len(attenuations)):

            f = log_interp_1d(energies[ii], attenuations[ii])
            attenuation_all[1] += f(energies_larger)

        # attenuation_all[1] *= densities[kk]
        np.savetxt(os.path.join(data_path, 'mu_over_rho', name+".csv"),
                   attenuation_all, fmt="%.8G", delimiter=",")
        logging.info(
            f'    Saved {name} atten to file in data/mu_over_rho/{name}.csv')
