import os
import numpy as np
import csv
import tempfile

# Set OpenEMS path
OPENEMS_PATH = r'C:\Users\sri\Downloads\openEMS_x64_v0.0.36-93-g7b9cd51_msvc\openEMS'
os.environ['OPENEMS_INSTALL_PATH'] = OPENEMS_PATH

# Required for Python 3.8+ on Windows to find DLLs
if hasattr(os, 'add_dll_directory'):
    os.add_dll_directory(OPENEMS_PATH)

from CSXCAD import ContinuousStructure
from openEMS import openEMS
from openEMS.physical_constants import *

# 1. Configuration (Targeting 5-8 GHz range for IoT)
# Increased density for better accuracy (4x4x4 = 64 samples)
VARIABLES = {
    "Length": np.linspace(10.0, 12.0, 4).tolist(),  # [10.0, 10.67, 11.33, 12.0]
    "Width":  np.linspace(14.0, 16.0, 4).tolist(),  # [14.0, 14.67, 15.33, 16.0]
    "FeedPos": np.linspace(2.5, 4.5, 4).tolist()    # [2.5, 3.17, 3.83, 4.5]
}

SUBSTRATE_EPS = 4.4        # FR4
SUBSTRATE_THICKNESS = 1.5  # mm
F_START, F_STOP = 3e9, 9e9 # Sweep 3 GHz to 9 GHz
F0, FC = (F_START + F_STOP) / 2, (F_STOP - F_START) / 2

def run_simulation(L, W, FP, sim_id):
    Sim_Path = os.path.join(tempfile.gettempdir(), f'openEMS_Patch_{sim_id}')
    if not os.path.exists(Sim_Path):
        os.makedirs(Sim_Path)

    FDTD = openEMS(NrTS=30000, EndCriteria=1e-4)
    FDTD.SetGaussExcite(F0, FC)
    FDTD.SetBoundaryCond(['MUR', 'MUR', 'MUR', 'MUR', 'MUR', 'MUR'])

    CSX = ContinuousStructure()
    FDTD.SetCSX(CSX)
    mesh = CSX.GetGrid()
    mesh.SetDeltaUnit(1e-3) # mm

    # Properties
    metal = CSX.AddMetal('Patch_Metal')
    substrate = CSX.AddMaterial('Substrate', epsilon=SUBSTRATE_EPS)
    gnd = CSX.AddMetal('GND')

    # Geometry
    # Patch (Top Layer)
    metal.AddBox([-L/2, -W/2, SUBSTRATE_THICKNESS], [L/2, W/2, SUBSTRATE_THICKNESS], priority=10)
    
    # Substrate Box (Slightly larger than patch)
    substrate.AddBox([-L/2-5, -W/2-5, 0], [L/2+5, W/2+5, SUBSTRATE_THICKNESS], priority=5)
    
    # Ground Plane
    gnd.AddBox([-L/2-5, -W/2-5, 0], [L/2+5, W/2+5, 0], priority=10)

    # --- Mesh Design ---
    mesh_res = C0 / F_STOP / 1e-3 / 15 # lambda/15
    mesh.AddLine('x', [-L/2-10, -L/2, L/2, L/2+10, FP])
    mesh.AddLine('y', [-W/2-10, -W/2, 0, W/2, W/2+10])
    mesh.AddLine('z', [-5, 0, SUBSTRATE_THICKNESS, SUBSTRATE_THICKNESS+10])
    mesh.SmoothMeshLines('all', mesh_res, 1.4)

    # --- Feed & Port ---
    # Probe feed from GND to Patch at offset FP
    port = FDTD.AddLumpedPort(1, 50, [FP, 0, 0], [FP, 0, SUBSTRATE_THICKNESS], 'z', 1.0, priority=50, edges2grid='xy')

    # NF2FF for Gain
    nf2ff = FDTD.CreateNF2FFBox(start=[-L/2-2, -W/2-2, -2], stop=[L/2+2, W/2+2, SUBSTRATE_THICKNESS+5])

    # Run
    FDTD.Run(Sim_Path, verbose=0, cleanup=True)

    # Post-process
    f = np.linspace(F_START, F_STOP, 801)
    port.CalcPort(Sim_Path, f)
    s11 = port.uf_ref / port.uf_inc
    s11_dB = 20.0 * np.log10(np.abs(s11))
    
    # Extract Bandwidth (S11 < -10dB)
    bw_indices = np.where(s11_dB < -10)[0]
    if len(bw_indices) > 0:
        # Find the widest continuous resonance
        segments = np.split(bw_indices, np.where(np.diff(bw_indices) != 1)[0] + 1)
        widest = max(segments, key=len)
        bandwidth = (f[widest[-1]] - f[widest[0]]) / 1e9 # GHz
        f_res = f[widest[np.argmin(s11_dB[widest])]]
        
        # Calculate Gain at Resonance
        try:
            theta = np.arange(0, 181, 10)
            phi = np.arange(0, 361, 10)
            nf2ff_res = nf2ff.CalcNF2FF(Sim_Path, f_res, theta, phi)
            max_gain = np.max(nf2ff_res.Dmax)
        except:
            max_gain = 0
    else:
        bandwidth, max_gain = 0, 0

    return bandwidth, max_gain

# 3. Main Execution
if __name__ == "__main__":
    csv_file = "dataset_patch.csv"
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Length_mm", "Width_mm", "FeedPos_mm", "Bandwidth_GHz", "Gain_dBi"])

        count = 0
        total_sims = len(VARIABLES["Length"]) * len(VARIABLES["Width"]) * len(VARIABLES["FeedPos"])
        for l in VARIABLES["Length"]:
            for w in VARIABLES["Width"]:
                for fp in VARIABLES["FeedPos"]:
                    count += 1
                    print(f"Simulating {count}/{total_sims}: L={l:.2f}, W={w:.2f}, FP={fp:.2f}...")
                    bw, gain = run_simulation(l, w, fp, count)
                    print(f"  Result: BW={bw:.2f}GHz, Gain={gain:.2f}dBi")
                    writer.writerow([l, w, fp, bw, gain])
                    f.flush()

    print(f"\nSuccess! Dataset saved to {csv_file}")
