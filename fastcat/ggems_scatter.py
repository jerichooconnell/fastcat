# A method to transform a fastcat simulation into a ggems simulation

import ggems as gg
import os

def run_ggems_scatter_simulation(phantom, detector_material, 
                                 nparticles, output_file='', output_dir=None,
                                 vis=False, angle=0,
                                 spectrum=None, 
                                 s_max=None,
                                 edep_detector=False):
      
    
    if spectrum is not None:
        kv_max = s_max
    else:
        kv_max = 100
    

    if output_dir is None:
        file_name = f'{output_file}ggems_{f"{nparticles:.0e}".replace("+", "")}_{kv_max:.0f}kVp_edep{edep_detector}'
    else:
        file_name = os.path.join(output_dir,f'{output_file}ggems_{f"{nparticles:.0e}".replace("+", "")}_{kv_max:.0f}kVp_edep{edep_detector}')
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
        opengl_manager.set_window_dimensions(window_dims[0], window_dims[1])
        opengl_manager.set_msaa(msaa)
        opengl_manager.set_background_color(window_color)
        opengl_manager.set_draw_axis(is_axis)
        opengl_manager.set_world_size(3.0, 3.0, 3.0, 'm')
        opengl_manager.set_image_output('data/axis')
        opengl_manager.set_displayed_particles(number_of_displayed_particles)
        opengl_manager.set_particle_color('gamma', 152, 251, 152)
        opengl_manager.initialize()

    # ------------------------------------------------------------------------------
    # STEP 3: Choosing an OpenCL device
    opencl_manager.set_device_to_activate(device)

    # ------------------------------------------------------------------------------
    # STEP 4: Setting GGEMS materials
    materials_database_manager.set_materials(phantom.material_file)

    # ------------------------------------------------------------------------------
    # STEP 5: Phantoms and systems

    if edep_detector:
        initialize_ct_edep_detector(phantom,detector_material,output_file)
    else:
        initialize_ct_counts_detector(phantom,detector_material,output_file)
        
    # Loading phantom
    ggphantom = gg.GGEMSVoxelizedPhantom('phantom')
    ggphantom.set_phantom(phantom.mhd_file, phantom.range_file)
    ggphantom.set_rotation(90., angle, 00.0, 'deg')
    ggphantom.set_position(0.0, 0.0, 0, 'mm')
    ggphantom.set_visible(True)
    # ggphantom.set_material_visible('Air', True)
    ggphantom.set_material_color('Water', color_name='blue')

    if get_phantom_dose:
        dosimetry = gg.GGEMSDosimetryCalculator()
        dosimetry.attach_to_navigator('phantom')
        dosimetry.set_output_basename(output_file)
        dosimetry.water_reference(False)
        dosimetry.set_tle(False)
        dosimetry.set_dosel_size(1, 1, 1, 'mm')
        dosimetry.uncertainty(True)
        dosimetry.edep(True)
        dosimetry.hit(True)
        dosimetry.edep_squared(False)

    # ------------------------------------------------------------------------------
    # STEP 6: Physics
    processes_manager.add_process('Compton', 'gamma', 'all')
    processes_manager.add_process('Photoelectric', 'gamma', 'all')
    processes_manager.add_process('Rayleigh', 'gamma', 'all')

    # Optional options, the following are by default
    processes_manager.set_cross_section_table_number_of_bins(220)
    processes_manager.set_cross_section_table_energy_min(1.0, 'keV')
    processes_manager.set_cross_section_table_energy_max(1.0, 'MeV')

    # ------------------------------------------------------------------------------
    # STEP 7: Cuts, by default but are 1 um
    range_cuts_manager.set_cut('gamma', 0.1, 'mm', 'all')
    range_cuts_manager.set_cut('e-', 0.1, 'mm', 'all')

    # ------------------------------------------------------------------------------
    # STEP 8: Source
    point_source = gg.GGEMSXRaySource('point_source')
    point_source.set_source_particle_type('gamma')
    point_source.set_number_of_particles(number_of_particles)
    point_source.set_position(-phantom.geomet.DSO, 0.0, 0.0, 'mm')
    point_source.set_rotation(0.0, 0.0, 0.0, 'deg')
    point_source.set_beam_aperture(12.5, 'deg')
    point_source.set_focal_spot_size(0.01, 0.01, 0, 'mm')

    if spectrum is not None:
        point_source.set_polyenergy(spectrum)
    else:
        point_source.set_monoenergy(100, 'keV')
        print('Using monoenergy at 100 keV')
    # point_source.set_polyenergy('data/spectrum_120kVp_2mmAl.dat')

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

    if is_draw_geom and is_gl: # Draw only geometry and do not run GGEMS
        opengl_manager.display()
    else: # Running GGEMS and draw particles
        ggems.run()

    # ------------------------------------------------------------------------------
    # STEP 10: Exit safely
    # dosimetry.delete()
    ggems.delete()
    gg.clean_safely()
    exit()

def initialize_ct_counts_detector(phantom,detector_material,output_file):

    cbct_detector = gg.GGEMSCTSystem('custom')
    cbct_detector.set_ct_type('flat')
    cbct_detector.set_number_of_modules(1, 1)
    a, b = phantom.geomet.nDetector
    d, e = phantom.geomet.dDetector
    cbct_detector.set_number_of_detection_elements(a, b, 1) # Might be a sham
    cbct_detector.set_size_of_detection_elements(d, e, 1, 'mm') # Still need to make this a module
    cbct_detector.set_material(detector_material)
    cbct_detector.set_source_detector_distance(phantom.geomet.DSD, 'mm') # Center of inside detector, adding half of detector (= SDD surface + 10.0/2 mm half of depth)
    cbct_detector.set_source_isocenter_distance(phantom.geomet.DSO, 'mm')
    cbct_detector.set_rotation(0.0, 0.0, 0.0, 'deg')
    cbct_detector.set_global_system_position(0.0, 0.0, 0.0, 'mm')
    cbct_detector.set_threshold(10, 'keV')
    cbct_detector.save(output_file)
    cbct_detector.store_scatter(True)
    cbct_detector.set_visible(True)

def initialize_ct_edep_detector(phantom,detector_material,output_file,thickness=1):

    volume_creator_manager = gg.GGEMSVolumeCreatorManager()
    a, b = phantom.geomet.nDetector
    d, e = phantom.geomet.dDetector
    volume_creator_manager.set_dimensions(1, a, b)
    volume_creator_manager.set_element_sizes(thickness, d, e, 'mm')
    volume_creator_manager.set_output('data/volume3.mhd')
    volume_creator_manager.set_range_output('data/range_volume3.txt')
    volume_creator_manager.set_material(detector_material)
    volume_creator_manager.set_data_type('MET_INT')
    volume_creator_manager.initialize()
    volume_creator_manager.write()

    # Loading phantom
    ggphantom = gg.GGEMSVoxelizedPhantom('phantom')
    ggphantom.set_phantom('data/volume3.mhd', 'data/range_volume3.txt')
    ggphantom.set_rotation(0.0, 0.0, 0.0, 'deg')
    ggphantom.set_position(phantom.geomet.DSD - phantom.geomet.DSO, 0.0, 0, 'mm')
    ggphantom.set_visible(True)

    dosimetry = gg.GGEMSDosimetryCalculator()
    dosimetry.attach_to_navigator('phantom')
    dosimetry.set_output_basename(output_file)
    dosimetry.water_reference(False)
    dosimetry.set_tle(False)
    dosimetry.set_dosel_size(thickness, d, e,  'mm')
    dosimetry.uncertainty(True)
    dosimetry.edep(True)
    dosimetry.hit(True)
    dosimetry.edep_squared(False)