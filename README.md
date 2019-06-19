gpu-vh (c) Dennis Bazow, Derek Everett

The following papers should be cited when referring to gpu-vh:
1) D. Bazow, U. W. Heinz and M. Strickland, Comput. Phys. Commun. 225, 92 (2018) doi:10.1016/j.cpc.2017.01.015
[arXiv:1608.06577 [physics.comp-ph]]
2) (L. Du) in preparation


## Purpose
gpu-vh is a code designed for the hydrodynamic simulation of heavy ion collisions.
Please see arXiv:1608.06577 for a description of the physics, as well as the KT algorithm which
is used to solve the hydrodynamic equations of motion.
This code is the GPU version of the algorithm described in this paper, and has been further optimized.

The up to date CPU version of this code can be found at https://github.com/derekeverett/cpu-vh
The two codes are designed to be as similar as possible, with the flexibility of running on heterogeneous computing platforms.

## Installation

To compile with cmake:
```
mkdir build & cd build
cmake ..
make
make install
```
There should now exist an executable gpu-vh in the parent directory.

## Usage

To run the code:
```
sh run.sh
```

All parameters can be set in the files inside the 'rhic-conf' directory.

'lattice.properties' contains parameters which determine the hydrodynamic grid size, number of points,
the time step size and number of time steps
NOTE* The number of points in x, y and eta need to be odd for the grid to be centered!

'ic.properties' contains the parameters for the initial conditions for hydrodynamics

'hydro.properties' contains parameters related to the viscous hydrodynamic evolution (for example the shear viscosity)

To run in ideal hydro mode commment out the macros PIMUNU and PI in DynamicalVariables.cuh.
To perform the Riemann problems, set the code to run in Cartesian coordinated by uncommenting the macro flag in SourceTerms.cu.
The configuration files for the different test problems are located in rhic/rhic-trunk/src/test/resources.
There is a flag in EquationOfState.cuh that allows you to switch between an ideal and QCD EoS.
The flux limiter parameter can be changed based on smooth or fluctuating initial conditions and is set in FluxLimiter.cu.

The freezeout surface file 'surface.dat' is written to the directory `output`, so this directory must exist when running gpu-vh.
The freezeout file is written in a format readable by the Cooper Frye and Sampling Code iS3D : https://github.com/derekeverett/iS3D
If running with initial conditions from file, the input files are read from the `input` directory.
