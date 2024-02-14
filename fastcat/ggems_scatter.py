# A method to transform a fastcat simulation into a ggems simulation
#
# The method takes as input a fastcat phantom object and a dictionary of simulation parameters
# and returns a ggems phantom object
#
# The simulation parameters are:
# detector_material: the material of the detector
# nparticles: the number of particles to simulate
# spectrum: the spectrum of the source
# s_max: the maximum energy of the spectrum
# edep_detector: whether or not to calculate the energy deposited in the detector
# dli: whether or not to make a dual layer detector
#
# The method also takes as input a dictionary of keyword arguments
# which are passed to the ggems simulation
#
# The method returns a ggems phantom object
#
# The method also saves the ggems phantom object to a pickle file
# and generates a bash script to run the ggems simulation
#
# The method also returns the name of the pickle file
#
# The method also returns the name of the bash script
#
# The method also returns the name of the output file
# ------------------------------------------------------------------------------


import pickle
import ggems as gg
import os


def save_ggems_simulation_parameters(phantom, output_dir, detector_material,
                                     nparticles, spectrum=None, s_max=None,
                                     edep_detector=False, dli=False, angles=None, **kwargs):
    '''
    Saves the parameters of the ggems simulation into a dictionary in the simulation
    object. This is used to save the parameters of the simulation to a pickle file
    '''
    phantom.angles = angles
    simulation_parameters = {}
    # simulation_parameters['output_file'] = output_file
    # simulation_parameters['output_dir'] = output_dir
    simulation_parameters['detector_material'] = detector_material
    simulation_parameters['nparticles'] = nparticles
    simulation_parameters['spectrum'] = spectrum
    simulation_parameters['s_max'] = s_max
    simulation_parameters['edep_detector'] = edep_detector
    simulation_parameters['dli'] = dli
    # simulation_parameters['phantom'] = phantom
    simulation_parameters['kwargs'] = kwargs

    output_file = os.path.join(
        output_dir, f'ggems_{f"{nparticles:.0e}".replace("+", "")}_{s_max:.0f}kVp')

    phantom.simulation_parameters = simulation_parameters

    # Create a new directory for the simulation
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Print the data to be saved
    print(f"Data to be saved: {simulation_parameters}")
    with open(os.path.join(output_file + '.pkl'), 'wb') as f:
        pickle.dump(phantom, f)
        print('Done saving simulation parameters to ' +
              os.path.join(output_file + '.pkl'))

    return output_file + '.pkl'


def generate_ggems_bash_script(phantom, output_dir, angles, detector_material,
                               nparticles, spectrum=None, s_max=None,
                               edep_detector=False, dli=False, **kwargs):
    '''
    Generate a bash script to give the scatter at angles
    '''

    fname = save_ggems_simulation_parameters(phantom, output_dir, detector_material,
                                             nparticles, spectrum, s_max, edep_detector, dli, angles=angles, **kwargs)

    # Write a bash script with each entry calling ggems_scatter script
    # with a different angle

    # get the directory path of the fastcat directory
    fastcat_dir = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(output_dir, 'ggems_scatter.sh'), 'w') as f:

        # Write the header
        f.write('#!/bin/bash\n\n')
        f.write(f'cd {output_dir} \n\n')

        # Write a for loop over the angles
        f.write(
            f'python {os.path.join(fastcat_dir,"fastcat","ggems_scatter_script.py")} {fname} --angle 0 --number 0 --flood_field True\n')
        for ii, angle in enumerate(angles):
            f.write(
                f'python {os.path.join(fastcat_dir,"fastcat","ggems_scatter_script.py")} {fname} --angle {angle} --number {ii}\n')

    phantom.bash_scipt = os.path.join(
        output_dir, 'ggems_scatter.sh')
    # Make the bash script executable
    os.system(f'chmod +x {phantom.bash_scipt}')


def run_ggems_scatter_simulation(phantom, detector_material,
                                 nparticles, output_file='',
                                 output_dir=None, vis=False, thickness=1,
                                 spectrum=None, angle=0, s_max=None,
                                 edep_detector=False, dli=False, **kwargs):

    if spectrum is not None:
        kv_max = s_max
    else:
        kv_max = 100

    # print(kwargs['flood_field'])
    if output_dir is None:
        if 'flood_field' in kwargs:
            if kwargs['flood_field']:
                file_name = f'ggems_{f"{nparticles:.0e}".replace("+", "")}_{kv_max:.0f}kVp_{output_file}_flood_field'
            else:
                file_name = f'ggems_{f"{nparticles:.0e}".replace("+", "")}_{kv_max:.0f}kVp_{output_file}'
        else:
            file_name = f'ggems_{f"{nparticles:.0e}".replace("+", "")}_{kv_max:.0f}kVp_{output_file}'
    else:
        if 'flood_field' in kwargs:
            if kwargs['flood_field']:
                file_name = os.path.join(
                    output_dir, f'ggems_{f"{nparticles:.0e}".replace("+", "")}_{kv_max:.0f}kVp_{output_file}_flood_field')
            else:
                file_name = os.path.join(
                    output_dir, f'ggems_{f"{nparticles:.0e}".replace("+", "")}_{kv_max:.0f}kVp_{output_file}')
        else:
            file_name = os.path.join(
                output_dir, f'ggems_{f"{nparticles:.0e}".replace("+", "")}_{kv_max:.0f}kVp_{output_file}')

    output_file = file_name

    # Getting arguments
    verbosity_level = 0
    seed = 777
    device = '0'
    number_of_particles = nparticles
    number_of_displayed_particles = 256
    is_axis = True
    msaa = int(8)
    window_color = 'black'
    window_dims = [800, 800]
    is_draw_geom = False
    is_gl = vis

    # ------------------------------------------------------------------------------
    # STEP 0: Level of verbosity during computation
    gg.GGEMSVerbosity(verbosity_level)

    # ------------------------------------------------------------------------------
    # STEP 1: Calling C++ singleton
    opencl_manager = gg.GGEMSOpenCLManager()
    materials_database_manager = gg.GGEMSMaterialsDatabaseManager()
    processes_manager = gg.GGEMSProcessesManager()
    range_cuts_manager = gg.GGEMSRangeCutsManager()
    # volume_creator_manager = GGEMSVolumeCreatorManager()

    # ------------------------------------------------------------------------------
    # STEP 2: Params for visualization
    if is_gl:
        opengl_manager = gg.GGEMSOpenGLManager()
        opengl_manager.set_window_dimensions(
            window_dims[0], window_dims[1])
        opengl_manager.set_msaa(msaa)
        opengl_manager.set_background_color(window_color)
        opengl_manager.set_draw_axis(is_axis)
        opengl_manager.set_world_size(3.0, 3.0, 3.0, 'm')
        opengl_manager.set_image_output('data/axis')
        opengl_manager.set_displayed_particles(
            number_of_displayed_particles)
        opengl_manager.set_particle_color(
            'gamma', 152, 251, 152)
        opengl_manager.initialize()

    # ------------------------------------------------------------------------------
    # STEP 3: Choosing an OpenCL device
    opencl_manager.set_device_to_activate(device)

    # ------------------------------------------------------------------------------
    # STEP 4: Setting GGEMS materials
    materials_database_manager.set_materials(
        phantom.material_file)

    # ------------------------------------------------------------------------------
    # STEP 5: Phantoms and systems

    if edep_detector:
        if dli:
            initialize_dli_edep(
                phantom, detector_material, output_file, thickness=thickness)
        else:
            initialize_ct_edep_detector(
                phantom, detector_material, output_file, thickness=thickness)
    else:
        initialize_ct_counts_detector(
            phantom, detector_material, output_file)

    # Loading phantom if flood field simulation not in kwaargs
    if 'flood_field' not in kwargs:
        ggphantom = gg.GGEMSVoxelizedPhantom('phantom')
        ggphantom.set_phantom(
            phantom.mhd_file, phantom.range_file)
        ggphantom.set_rotation(3.1415/2, angle, 0.0, 'rad')
        ggphantom.set_position(0.0, 0.0, 0, 'mm')
        ggphantom.set_visible(True)
        # ggphantom.set_material_visible('Air', True)
        ggphantom.set_material_color(
            'Water', color_name='blue')
    elif kwargs['flood_field']:
        pass
    else:
        ggphantom = gg.GGEMSVoxelizedPhantom('phantom')
        ggphantom.set_phantom(
            phantom.mhd_file, phantom.range_file)
        ggphantom.set_rotation(3.1415/2, angle, 00.0, 'rad')
        ggphantom.set_position(0.0, 0.0, 0, 'mm')
        ggphantom.set_visible(True)
        ggphantom.set_material_visible('Air', True)
        ggphantom.set_material_color(
            'Water', color_name='blue')

    # ------------------------------------------------------------------------------
    # STEP 6: Physics
    processes_manager.add_process('Compton', 'gamma', 'all')
    processes_manager.add_process(
        'Photoelectric', 'gamma', 'all')
    processes_manager.add_process(
        'Rayleigh', 'gamma', 'all')

    # Optional options, the following are by default
    processes_manager.set_cross_section_table_number_of_bins(
        220)
    processes_manager.set_cross_section_table_energy_min(
        1.0, 'keV')
    processes_manager.set_cross_section_table_energy_max(
        1.0, 'MeV')

    # ------------------------------------------------------------------------------
    # STEP 7: Cuts, by default but are 1 um
    range_cuts_manager.set_cut('gamma', 0.1, 'mm', 'all')
    range_cuts_manager.set_cut('e-', 0.1, 'mm', 'all')

    # ------------------------------------------------------------------------------
    # STEP 8: Source
    point_source = gg.GGEMSXRaySource('point_source')
    point_source.set_source_particle_type('gamma')

    if 'flood_field' not in kwargs:
        point_source.set_number_of_particles(
            number_of_particles)
    elif kwargs['flood_field']:
        print(kwargs['flood_field'])
        point_source.set_number_of_particles(
            10000000)  # 1e6 is good enough for the flood field
    else:
        point_source.set_number_of_particles(
            number_of_particles)
    point_source.set_position(-phantom.geomet.DSO,
                              0.0, 0.0, 'mm')
    point_source.set_rotation(0.0, 0.0, 0.0, 'deg')

    if 'bowtie_file' in kwargs:
        point_source.read_bowtie_file(kwargs['bowtie_file'])

    point_source.set_beam_aperture(12.5, 'deg')
    point_source.set_focal_spot_size(0.01, 0.01, 0, 'mm')

    if spectrum is not None:
        point_source.set_polyenergy(spectrum)
    else:
        point_source.set_monoenergy(100, 'keV')
        print('Using monoenergy at 100 keV')
    # point_source.set_polyenergy('data/spectrum_120kVp_2mmAl.dat')

    # Check if the keywoard 'return_dose' is in kwargs
    if 'return_dose' in kwargs:
        if kwargs['return_dose']:
            initialize_phantom_dose(output_file, phantom)

    # ------------------------------------------------------------------------------
    # STEP 9: GGEMS simulation
    ggems = gg.GGEMS()
    ggems.opencl_verbose(False)
    ggems.material_database_verbose(False)
    ggems.navigator_verbose(False)
    ggems.source_verbose(True)
    ggems.memory_verbose(True)
    ggems.process_verbose(True)
    ggems.range_cuts_verbose(True)
    ggems.random_verbose(True)
    ggems.profiling_verbose(True)
    ggems.tracking_verbose(False, 0)

    # Initializing the GGEMS simulation
    ggems.initialize(seed)

    if is_draw_geom and is_gl:  # Draw only geometry and do not run GGEMS
        opengl_manager.display()
    else:  # Running GGEMS and draw particles
        ggems.run()

    # ------------------------------------------------------------------------------
    # STEP 10: Exit safely
    # dosimetry.delete()
    ggems.delete()
    gg.clean_safely()
    exit()


def initialize_phantom_dose(output_file, phantom):

    dosimetry = gg.GGEMSDosimetryCalculator()
    dosimetry.attach_to_navigator('phantom')
    dosimetry.set_output_basename(
        output_file + '_phantom_dose')
    dosimetry.water_reference(False)
    dosimetry.set_tle(False)
    dosimetry.set_dosel_size(
        phantom.sVoxel[0], phantom.sVoxel[1], phantom.sVoxel[2], 'mm')
    dosimetry.uncertainty(True)
    dosimetry.edep(False)
    dosimetry.hit(False)
    dosimetry.edep_squared(False)


def initialize_ct_counts_detector(phantom, detector_material, output_file):

    cbct_detector = gg.GGEMSCTSystem('custom')
    cbct_detector.set_ct_type('flat')
    cbct_detector.set_number_of_modules(1, 1)
    a, b = phantom.geomet.nDetector
    d, e = phantom.geomet.dDetector
    cbct_detector.set_number_of_detection_elements(
        a, b, 1)  # Might be a sham
    cbct_detector.set_size_of_detection_elements(
        d, e, 1, 'mm')  # Still need to make this a module
    cbct_detector.set_material(detector_material)
    # Center of inside detector, adding half of detector (= SDD surface + 10.0/2 mm half of depth)
    cbct_detector.set_source_detector_distance(
        phantom.geomet.DSD, 'mm')
    cbct_detector.set_source_isocenter_distance(
        phantom.geomet.DSO, 'mm')
    cbct_detector.set_rotation(0.0, 0.0, 0.0, 'deg')
    cbct_detector.set_global_system_position(
        0.0, 0.0, 0.0, 'mm')
    cbct_detector.set_threshold(10, 'keV')
    cbct_detector.save(output_file)
    cbct_detector.store_scatter(True)
    cbct_detector.set_visible(True)


def initialize_ct_edep_detector(phantom, detector_material, output_file, thickness=1):

    volume_creator_manager = gg.GGEMSVolumeCreatorManager()
    a, b = phantom.geomet.nDetector
    d, e = phantom.geomet.dDetector
    volume_creator_manager.set_dimensions(1, a, b)
    volume_creator_manager.set_element_sizes(
        thickness, d, e, 'mm')
    volume_creator_manager.set_output('data/volume3.mhd')
    volume_creator_manager.set_range_output(
        'data/range_volume3.txt')
    volume_creator_manager.set_material(detector_material)
    volume_creator_manager.set_data_type('MET_INT')
    volume_creator_manager.initialize()
    volume_creator_manager.write()

    # Loading phantom
    ggphantom = gg.GGEMSVoxelizedPhantom('phantom1')
    ggphantom.set_phantom(
        'data/volume3.mhd', 'data/range_volume3.txt')
    ggphantom.set_rotation(0.0, 0.0, 0.0, 'deg')
    ggphantom.set_position(
        phantom.geomet.DSD - phantom.geomet.DSO, 0.0, 0, 'mm')
    ggphantom.set_visible(True)
    # ggphantom = gg.GGEMSVoxelizedPhantom('phantom1')
    # ggphantom.set_phantom(
    #     'data/volume3.mhd', 'data/range_volume3.txt')
    # ggphantom.set_rotation(0.0, 0.0, 0.0, 'deg')
    # ggphantom.set_position(
    #     phantom.geomet.DSD - phantom.geomet.DSO, 0.0, 0, 'mm')
    # ggphantom.set_visible(True)

    dosimetry = gg.GGEMSDosimetryCalculator()
    dosimetry.attach_to_navigator('phantom1')
    dosimetry.set_output_basename(output_file)
    dosimetry.water_reference(False)
    dosimetry.set_tle(False)
    dosimetry.set_dosel_size(thickness, d, e,  'mm')
    dosimetry.uncertainty(True)
    dosimetry.edep(True)
    dosimetry.hit(True)
    dosimetry.edep_squared(False)


def initialize_dli_edep(phantom, detector_material, output_file, thickness=1):

    volume_creator_manager = gg.GGEMSVolumeCreatorManager()
    a, b = phantom.geomet.nDetector
    d, e = phantom.geomet.dDetector
    volume_creator_manager.set_dimensions(1, a, b)
    volume_creator_manager.set_element_sizes(
        thickness, d, e, 'mm')
    volume_creator_manager.set_output('data/volume3.mhd')
    volume_creator_manager.set_range_output(
        'data/range_volume3.txt')
    volume_creator_manager.set_material(detector_material)
    volume_creator_manager.set_data_type('MET_INT')
    volume_creator_manager.initialize()
    volume_creator_manager.write()

    # Loading phantom
    ggphantom = gg.GGEMSVoxelizedPhantom('phantom1')
    ggphantom.set_phantom(
        'data/volume3.mhd', 'data/range_volume3.txt')
    ggphantom.set_rotation(0.0, 0.0, 0.0, 'deg')
    ggphantom.set_position(
        phantom.geomet.DSD - phantom.geomet.DSO, 0.0, 0.0, 'mm')
    ggphantom.set_visible(True)

    dosimetry = gg.GGEMSDosimetryCalculator()
    dosimetry.attach_to_navigator('phantom1')
    dosimetry.set_output_basename(output_file + '1')
    dosimetry.water_reference(False)
    dosimetry.set_tle(False)
    dosimetry.set_dosel_size(thickness, d, e,  'mm')
    dosimetry.uncertainty(True)
    dosimetry.edep(True)
    dosimetry.hit(True)
    dosimetry.edep_squared(False)

    volume_creator_manager2 = gg.GGEMSVolumeCreatorManager()
    a, b = phantom.geomet.nDetector
    d, e = phantom.geomet.dDetector
    volume_creator_manager2.set_dimensions(1, a, b)
    volume_creator_manager2.set_element_sizes(
        thickness, d, e, 'mm')
    volume_creator_manager2.set_output('data/volume4.mhd')
    volume_creator_manager2.set_range_output(
        'data/range_volume4.txt')
    volume_creator_manager2.set_material(detector_material)
    volume_creator_manager2.set_data_type('MET_INT')
    volume_creator_manager2.initialize()
    volume_creator_manager2.write()

    # Loading phantom
    ggphantom2 = gg.GGEMSVoxelizedPhantom('phantom2')
    ggphantom2.set_phantom(
        'data/volume4.mhd', 'data/range_volume4.txt')
    ggphantom2.set_rotation(0.0, 0.0, 0.0, 'deg')
    ggphantom2.set_position(
        phantom.geomet.DSD - phantom.geomet.DSO + 1.5, 0.0, 0.0, 'mm')
    ggphantom2.set_visible(True)

    dosimetry2 = gg.GGEMSDosimetryCalculator()
    dosimetry2.attach_to_navigator('phantom2')
    dosimetry2.set_output_basename(output_file + '2')
    dosimetry2.water_reference(False)
    dosimetry2.set_tle(False)
    dosimetry2.set_dosel_size(thickness, d, e,  'mm')
    dosimetry2.uncertainty(True)
    dosimetry2.edep(True)
    dosimetry2.hit(True)
    dosimetry2.edep_squared(False)
