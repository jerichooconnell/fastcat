# Test the fastcat mhd file reader from fastcat.utils
import fastcat as fc
import argparse
import fastcat.ggems_scatter as gg_scatter
import pickle
import os

# ------------------------------------------------------------------------------
# Read arguments
parser = argparse.ArgumentParser(
    prog='image_viewer.py',
    description='''-->> 6 - OpenGL Visualization Example <<--''',
    epilog='''''',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# takes as input the name of a pickle file containing the simulation parameters
# and an angle to rotate the simulation by

parser.add_argument('filename', type=str,
                    help='''Name of the file to read''')
parser.add_argument('--angle', type=float, default=0.0,
                    help='''Angle to rotate the simulation by''')
parser.add_argument('--number', type=int, default=0,
                    help='''Number to append to the output file''')
parser.add_argument('--flood_field', type=bool, default=False,
                    help='''Whether or not to generate a flood field''')

args = parser.parse_args()


def ggems_scatter_from_phantom_file(phantom_file, angle=0, output_suffix='', flood_field=False):
    '''
    Loads the parameters of the ggems simulation into a dictionary in the simulation
    object. This is used to save the parameters of the simulation to a pickle file
    '''
    with open(phantom_file, 'rb') as f:
        phantom = pickle.load(f)
        print(
            'Done loading simulation parameters from ' + phantom_file)

    # Get the simulation parameters
    simulation_parameters = phantom.simulation_parameters
    detector_material = simulation_parameters['detector_material']
    nparticles = simulation_parameters['nparticles']
    spectrum = simulation_parameters['spectrum']
    s_max = simulation_parameters['s_max']
    edep_detector = simulation_parameters['edep_detector']
    dli = simulation_parameters['dli']
    kwargs = simulation_parameters['kwargs']

    # add flood field to kwargs
    kwargs['flood_field'] = flood_field

    # Check if the output directory exists
    if not os.path.exists('out'):
        os.makedirs('out')

    # Run the ggems simulation
    gg_scatter.run_ggems_scatter_simulation(phantom, detector_material,
                                            nparticles, spectrum=spectrum, s_max=s_max, output_file=output_suffix,
                                            edep_detector=edep_detector, output_dir='out', dli=dli, angle=angle, **kwargs)


# Make the number have four digits
number = str(args.number).zfill(4)
print('Running ggems scatter simulation from ' +
      args.filename + number)
ggems_scatter_from_phantom_file(
    args.filename, angle=args.angle, output_suffix=number,  flood_field=args.flood_field)
