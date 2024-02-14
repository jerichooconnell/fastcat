import pickle
import numpy as np
import os


def write_fastmc_xml_file(phantom, sim_dir, out_dir, angle=0, half_fan=False, **kwargs):

    if half_fan:
        print('Half fan mode')
        ShiftXcm = 0
        ShiftYcm = -16
    else:
        print('Full fan mode')
        ShiftXcm = 0
        ShiftYcm = 0
    # make the sim and out directories if they don't exist
    if not os.path.exists(sim_dir):
        os.makedirs(sim_dir)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for angle in phantom.sim_angles:
        fname = os.path.join(
            sim_dir, f'fastmc_{np.rad2deg(angle):.2f}.fmc')
        output_file = os.path.join(
            out_dir, f'fastmc_{np.rad2deg(angle):.2f}')
        xml_modified = f'''<?xml version="1.0" encoding="UTF-8"?>
<FastMCParameters>
    <parameter name="NumberOfParticles">{phantom.nparticles_per_angle}</parameter>
    <parameter name="XRayTubeAngle">{np.rad2deg(angle)}</parameter>
    <parameter name="SourceDetectorDistance">{phantom.geomet.DSD}</parameter>
    <parameter name="SourceIsocenterDistance">{phantom.geomet.DSO}</parameter>
    <parameter name="FanAngle">-1</parameter>
    <parameter name="DetectorElementsX">{phantom.geomet.nDetector[0]}</parameter>
    <parameter name="DetectorElementsY">{phantom.geomet.nDetector[1]}</parameter>
    <parameter name="ElementXSize">{phantom.geomet.dDetector[0]}</parameter>
    <parameter name="ElementYSize">{phantom.geomet.dDetector[1]}</parameter>
    <parameter name="DetectorThickness">{phantom.detector_thickness}</parameter>
    <parameter name="DetectorThickness2">{phantom.detector_thickness2}</parameter>
    <parameter name="DetectorMaterial">{phantom.detector_material}</parameter>
    <parameter name="FieldX1At100cm">{np.round(phantom.geomet.sDetector[0]/3 + ShiftXcm*10)}</parameter>
    <parameter name="FieldX2At100cm">{np.round(phantom.geomet.sDetector[0]/3 - ShiftXcm*10)}</parameter>
    <parameter name="FieldY1At100cm">{np.round(phantom.geomet.sDetector[1]/3 + ShiftYcm*10)}</parameter>
    <parameter name="FieldY2At100cm">{np.round(phantom.geomet.sDetector[1]/3 - ShiftYcm*10)}</parameter>
    <parameter name="ShiftXcm">{ShiftXcm}</parameter>
    <parameter name="ShiftYcm">{ShiftYcm}</parameter>
    <parameter name="BowtieFile">{phantom.bowtie_file}</parameter>
    <parameter name="OutputFilename">{output_file}</parameter>
    <parameter name="MaterialsDatabase">{phantom.material_file}</parameter>
    <parameter name="SpectrumFile">{phantom.spectrum_file}</parameter>
    <parameter name="PhantomFile">{phantom.mhd_file}</parameter>
    <parameter name="PhantomRange">{phantom.range_file}</parameter>
    <parameter name="PhantomZpos">0</parameter>
</FastMCParameters>'''

        with open(fname, 'w') as f:
            f.write(xml_modified)


def write_fastmc_flood_field_xml_file(phantom, sim_dir, out_dir, angle=0, half_fan=False, nphot=int(1e10), **kwargs):

    if half_fan:
        print('Half fan mode')
        ShiftXcm = 0
        ShiftYcm = -16
    else:
        print('Full fan mode')
        ShiftXcm = 0
        ShiftYcm = 0
    # make the sim and out directories if they don't exist
    if os.path.exists(sim_dir):
        os.system(f'rm -rf {sim_dir}')
    os.makedirs(sim_dir)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    # If the sim directory exists then delete it

    angle = 0
    fname = os.path.join(
        sim_dir, f'fastmc_00.0_flood.fmc')
    output_file = os.path.join(
        out_dir, f'fastmc_00.0_flood')
    xml_modified = f'''<?xml version="1.0" encoding="UTF-8"?>
<FastMCParameters>
    <parameter name="NumberOfParticles">{nphot}</parameter>
    <parameter name="XRayTubeAngle">{angle}</parameter>
    <parameter name="SourceDetectorDistance">{phantom.geomet.DSD}</parameter>
    <parameter name="SourceIsocenterDistance">{phantom.geomet.DSO}</parameter>
    <parameter name="FanAngle">-1</parameter>
    <parameter name="DetectorElementsX">{phantom.geomet.nDetector[0]}</parameter>
    <parameter name="DetectorElementsY">{phantom.geomet.nDetector[1]}</parameter>
    <parameter name="ElementXSize">{phantom.geomet.dDetector[0]}</parameter>
    <parameter name="ElementYSize">{phantom.geomet.dDetector[1]}</parameter>
    <parameter name="DetectorThickness">{phantom.detector_thickness}</parameter>
    <parameter name="DetectorThickness2">{phantom.detector_thickness2}</parameter>
    <parameter name="DetectorMaterial">{phantom.detector_material}</parameter>
    <parameter name="FieldX1At100cm">{phantom.geomet.sDetector[0]/3 + ShiftXcm*10}</parameter>
    <parameter name="FieldX2At100cm">{phantom.geomet.sDetector[0]/3 - ShiftXcm*10}</parameter>
    <parameter name="FieldY1At100cm">{phantom.geomet.sDetector[1]/3 + ShiftYcm*10}</parameter>
    <parameter name="FieldY2At100cm">{phantom.geomet.sDetector[1]/3 - ShiftYcm*10}</parameter>
    <parameter name="ShiftXcm">{ShiftXcm}</parameter>
    <parameter name="ShiftYcm">{ShiftYcm}</parameter>
    <parameter name="BowtieFile">{phantom.bowtie_file}</parameter>
    <parameter name="OutputFilename">{output_file}</parameter>
    <parameter name="MaterialsDatabase">{phantom.material_file}</parameter>
    <parameter name="SpectrumFile">{phantom.spectrum_file}</parameter>
</FastMCParameters>'''

    phantom.nparticles_flood = nphot

    with open(fname, 'w') as f:
        f.write(xml_modified)

    output_file2 = os.path.join(
        kwargs['file_name'], f'fastmc_{f"{phantom.nparticles_per_angle:.0e}".replace("+", "")}_{len(phantom.sim_angles):.0f}angles')

    # Print the data to be saved
    print(f"Data to be saved:")
    with open(os.path.join(output_file2 + '.pkl'), 'wb') as f:
        pickle.dump(phantom, f)
        print('Done saving simulation parameters to ' +
              os.path.join(output_file + '.pkl'))


def run_fastmc_files(lib_path, sim_dir):

    # Find all the fastmc files in the sim_dir
    files = [f for f in os.listdir(
        sim_dir) if f.endswith('.fmc')]

    # Sort the files by angle
    files.sort(key=lambda x: float(
        x.split('_')[1].split('.')[0]))

    # Run the fastmc files individually
    for f in files:
        os.system(f'{lib_path} {os.path.join(sim_dir, f)}')
