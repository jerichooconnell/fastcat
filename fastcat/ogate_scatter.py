import opengate as gate
import opengate.tests.utility as utility
from scipy.spatial.transform import Rotation
import os

def ggems_material_file_2_gate_material_file(ggems_material_file):  
    '''Add a couple things to the gate material file'''
    with open(ggems_material_file,'r') as f:
        lines = f.readlines()

    # Add the header
    lines.insert(0,'[Materials]\n')

def run_ogate_scatter_simulation(phantom, detector_material,material_file, 
                                 nparticles, output_file = '', output_dir=None, spectrum=None,
                                 vis=False, nt = 1):
    
    if spectrum is not None:
        kv_max = spectrum.x.max()
    else:
        kv_max = 100
    

    if output_dir is None:
        file_name = f'{output_file}ogate_{f"{nparticles:.0e}".replace("+", "")}_{kv_max:.0f}kVp'
    else:
        file_name = os.path.join(output_dir,f'{output_file}ogate_{f"{nparticles:.0e}".replace("+", "")}_{kv_max:.0f}kVp')
    output_file = file_name
    
    paths = utility.get_default_test_paths(
        __file__, "gate_test004_simulation_stats_actor"
    )

    # create the simulation
    sim = gate.Simulation()

    # main options
    ui = sim.user_info
    ui.g4_verbose = True
    ui.g4_verbose_level = 1
    ui.visu = vis
    ui.visu_type = "vrml"
    ui.visu_filename = "geant4VisuFile.wrl"
    ui.visu_verbose = True
    ui.number_of_threads = nt
    ui.random_engine = "MersenneTwister"
    ui.random_seed = "auto"
    # ui.running_verbose_level = gate.logger.EVENT

    # change physics
    p = sim.get_physics_user_info()
    p.physics_list_name = "G4EmStandardPhysics_option4"
    # p.enable_decay = True

    m = gate.g4_units.m
    um = gate.g4_units.um

    # Set this really high to cut them all
    # sim.physics_manager.global_production_cuts.electron = 1 * m


    sim.physics_manager.global_production_cuts.gamma = 100 * um
    sim.physics_manager.global_production_cuts.positron = 100 * um
    sim.physics_manager.global_production_cuts.proton = 100 * um

    # Units
    m = gate.g4_units.m
    MeV = gate.g4_units.MeV
    mm = gate.g4_units.mm
    keV = gate.g4_units.keV
    deg = gate.g4_units.deg

    # add a material database
    sim.add_material_database(material_file)

    # Change world size
    world = sim.world
    world.size = [3 * m, 3 * m, 3 * m]
    world.material = "G4_AIR"

    a, b = phantom.geomet.nDetector
    d, e = phantom.geomet.dDetector
    f, g = phantom.geomet.sDetector

    # detector in w2 (on top of world)
    det = sim.add_volume("Box", "detector")
    det.mother = "world"
    det.material = detector_material
    det.size = [ f * mm, 1 * mm, g * mm]
    det.translation = [0,-(phantom.geomet.DSD - phantom.geomet.DSO) * mm, 0]
    det.color = [1, 0, 0, 1]

    # Add a pencil beam source
    source = sim.add_source("GenericSource", "source")
    # source.energy.mono = 0.1 * MeV
    source.particle = "gamma"
    source.position.type = "box"
    # Give theta and phi that are  centered around the z axis
    
    source.direction.theta = [(90 - 7.8) * deg, (90 + 7.8) * deg] # Box
    source.direction.phi = [(90 - 7.8) * deg, (90 + 7.8) * deg] 
    source.direction.type = "iso"
    source.position.size = [0.01 * mm, 0.01 * mm, 0.01 * mm]
    source.position.translation = [0, phantom.geomet.DSO * mm, 0]
    source.n = int(nparticles / (nt)) 
    # source.rotation = Rotation.from_euler('x', 90, degrees=True).as_matrix()


    if spectrum is not None:
        w = spectrum.y
        en = spectrum.x

        for ii, value in enumerate(en):
            en[ii] = value * keV

        source.energy.type = "histogram"
        source.energy.histogram_weight = w
        source.energy.histogram_energy = en
    else:
        source.energy.mono = 0.1 * MeV

    # add phsp actor detector 1 (overlap!)
    # add phsp actor detector 1 (overlap!)
    dose = sim.add_actor("DoseActor", "edep")
    dose.output = output_file + '.mhd'
    dose.mother = det.name
    dose.size = [ a, 1, b]
    dose.spacing = [ d * mm, 1 * mm, e * mm]
    dose.hit_type = "random"
    dose.uncertainty = True

    if vis == False:
        patient = sim.add_volume("Image", "patient")
        patient.image = phantom.mhd_file
        patient.mother = "world"
        patient.rotation = Rotation.from_euler('zy', [270,180], degrees=True).as_matrix()
        # patient.rotation = Rotation.from_euler('xz', [180,90], degrees=True).as_matrix()
        # patient.rotation = Rotation.from_euler('y', 90, degrees=True).as_matrix()
        # patient.material = "G4_AIR"  # material used by default

        # Take the materials and the indecis from the phan_map and assign them to ranges
        voxel_materials = []
        for ii, material in enumerate(phantom.phan_map):
            voxel_materials.append([ii - 0.25, ii + 0.25, material])
        patient.voxel_materials = voxel_materials

    # sim.physics_manager.set_cut("detector", "all", 0.01 * mm)

    # print
    print("Geometry trees: ")
    print(sim.dump_tree_of_volumes())

    # start simulation
    output = sim.start()
    # print('Number of particles: {}'.format(int(nparticles / (nt))))