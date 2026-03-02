# Modified 3D-2PT Framework for Ionic Hydration Entropy

This repository contains the analysis scripts and sample data for the paper:  
**"Unveiling the Hidden Role of Outer-Shell Water in ionic hydration Thermodynamics"**

## Overview
This code extends the traditional Three-Dimensional Two-Phase Thermodynamics (3D-2PT) method by introducing a **"Confined Rigid Rotor"** model. It explicitly accounts for the rotational symmetry breaking of water molecules induced by the long-range anisotropic electrostatic field of ions, allowing for an accurate, spatially resolved calculation of ionic hydration entropy.

## Features
- Parsing LAMMPS molecular dynamics trajectories.
- Construction of 3D spatial grids ($48 \times 48 \times 48$ voxels).
- Calculation of localized translational and rotational Vibrational Density of States (VDOS).
- Correction for gas-phase rotational entropy using the Langevin function-based orientational restriction model.

## Prerequisites
- Python 3.8+ 
- NumPy, SciPy, MDAnalysis，Numba 

## Usage
1. **Prepare Trajectory**: Ensure your LAMMPS trajectory is saved with high frequency (e.g., every 4 fs).
2. **Run Analysis**: 
   ```bash
   python calculate_3d2pt.py --traj md_run.xtc --topo system.data --ion_index 1
