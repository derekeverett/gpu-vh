/*
* HydroPlugin.c
*
*  Created on: Oct 23, 2015
*      Author: bazow
*/

#include <stdlib.h>
#include <stdio.h> // for printf
#include <cmath>

// for timing
#include <ctime>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>

//for cornelius and writing freezeout file
#include <fstream>
#include "cornelius-c++-1.3/cornelius.cpp"
#include "FreezeOut.cpp"
#include "Memory.cpp"

#include "edu/osu/rhic/harness/hydro/HydroPlugin.h"
#include "edu/osu/rhic/trunk/hydro/DynamicalVariables.cuh"
#include "edu/osu/rhic/harness/lattice/LatticeParameters.h"
#include "edu/osu/rhic/harness/ic/InitialConditionParameters.h"
#include "edu/osu/rhic/harness/hydro/HydroParameters.h"
#include "edu/osu/rhic/harness/io/FileIO.h"
#include "edu/osu/rhic/trunk/ic/InitialConditions.h"
#include "edu/osu/rhic/trunk/hydro/FullyDiscreteKurganovTadmorScheme.cuh"
#include "edu/osu/rhic/harness/init/CudaConfiguration.cuh"
#include "edu/osu/rhic/trunk/hydro/EnergyMomentumTensor.cuh"
#include "edu/osu/rhic/trunk/eos/EquationOfState.cuh"
#include "edu/osu/rhic/trunk/hydro/GhostCells.cuh"
#include "edu/osu/rhic/trunk/hydro/HydrodynamicValidity.cuh"

#define FREQ 100
#define FOFREQ 10 //call freezeout surface finder every FOFREQ timesteps
#define FOTEST 0 //if true, freezeout surface file is written with proper times rounded (down) to step size
#define FOFORMAT 0 // 0 : write f.o. surface to ASCII file ;  1 : write to binary file

void outputDynamicalQuantities(double t, const char *outputDir, void * latticeParams) {
	output(e, t, outputDir, "e", latticeParams);
	//	/*
	output(p, t, outputDir, "p", latticeParams);
	output(u->ux, t, outputDir, "ux", latticeParams);
	output(u->uy, t, outputDir, "uy", latticeParams);
	output(u->un, t, outputDir, "un", latticeParams);
	output(u->ut, t, outputDir, "ut", latticeParams);
	//output(q->ttt, t, outputDir, "ttt", latticeParams);
	//output(q->ttn, t, outputDir, "ttn", latticeParams);
	#ifdef PIMUNU
	///*
	output(q->pixx, t, outputDir, "pixx", latticeParams);
	output(q->pixy, t, outputDir, "pixy", latticeParams);
	output(q->pixn, t, outputDir, "pixn", latticeParams);
	output(q->piyy, t, outputDir, "piyy", latticeParams);
	output(q->piyn, t, outputDir, "piyn", latticeParams);

	output(q->pitt, t, outputDir, "pitt", latticeParams);
	output(q->pitx, t, outputDir, "pitx", latticeParams);
	output(q->pity, t, outputDir, "pity", latticeParams);
	output(q->pitn, t, outputDir, "pitn", latticeParams);
	output(q->pinn, t, outputDir, "pinn", latticeParams);
	#endif
	#ifdef PI
	output(q->Pi, t, outputDir, "Pi", latticeParams);
	#endif
	///*
	//output(validityDomain->regulations, t, outputDir, "regulations", latticeParams);
	//output(validityDomain->inverseReynoldsNumberPimunu, t, outputDir, "Rpi", latticeParams);
	//output(validityDomain->inverseReynoldsNumberTilde2Pimunu, t, outputDir, "R2pi", latticeParams);
	//output(validityDomain->inverseReynoldsNumberPi, t, outputDir, "RPi", latticeParams);
	//output(validityDomain->inverseReynoldsNumberTilde2Pi, t, outputDir, "R2Pi", latticeParams);
	//output(validityDomain->knudsenNumberTaupi, t, outputDir, "KnTaupi", latticeParams);
	//output(validityDomain->knudsenNumberTauPi, t, outputDir, "KnTauPi", latticeParams);
	// for debugging purposes
	//output(validityDomain->taupi, t, outputDir, "taupi", latticeParams);
	//output(validityDomain->dxux, t, outputDir, "dxux", latticeParams);
	//output(validityDomain->dyuy, t, outputDir, "dyuy", latticeParams);
	//output(validityDomain->theta, t, outputDir, "theta", latticeParams);
	//*/
}

#define CLOCKS_PER_MILLISEC (CLOCKS_PER_SEC / 1000)
class Stopwatch {
private:
	time_t start, end;
public:
	Stopwatch() {
		start = clock();
		end = 0;
	}
	void tic() {
		start = clock();
	}
	void toc() {
		end = clock();
	}
	double elapsedTime() {
		return ((double) (end - start)) / CLOCKS_PER_MILLISEC;
	}
};

void run(void * latticeParams, void * initCondParams, void * hydroParams, const char *rootDirectory, const char *outputDir) {
	struct LatticeParameters * lattice = (struct LatticeParameters *) latticeParams;
	struct InitialConditionParameters * initCond = (struct InitialConditionParameters *) initCondParams;
	struct HydroParameters * hydro = (struct HydroParameters *) hydroParams;

	/************************************************************************************\
	* System configuration (C/CUDA)
	/************************************************************************************/
	int nt = lattice->numProperTimePoints;
	int nx = lattice->numLatticePointsX;
	int ny = lattice->numLatticePointsY;
	int nz = lattice->numLatticePointsRapidity;
	int ncx = lattice->numComputationalLatticePointsX;
	int ncy = lattice->numComputationalLatticePointsY;
	int ncz = lattice->numComputationalLatticePointsRapidity;
	int nElements = ncx * ncy * ncz;

	double t0 = hydro->initialProperTimePoint;
	double dt = lattice->latticeSpacingProperTime;
	double dx = lattice->latticeSpacingX;
	double dy = lattice->latticeSpacingY;
	double dz = lattice->latticeSpacingRapidity;
	double e0 = initCond->initialEnergyDensity;

	double freezeoutTemperatureGeV = hydro->freezeoutTemperatureGeV;
	const double hbarc = 0.197326938;
	const double freezeoutTemperature = freezeoutTemperatureGeV/hbarc;
	//	const double freezeoutEnergyDensity = e0*pow(freezeoutTemperature,4);
	const double freezeoutEnergyDensity = equilibriumEnergyDensity(freezeoutTemperature);
	printf("Grid size = %d x %d x %d\n", nx, ny, nz);
	printf("spatial resolution = (%.3f, %.3f, %.3f)\n", lattice->latticeSpacingX, lattice->latticeSpacingY, lattice->latticeSpacingRapidity);
	printf("freezeout temperature = %.3f [fm^-1] (eF = %.3f [fm^-4])\n", freezeoutTemperature, freezeoutEnergyDensity);

	printf("eta/s = %.6f\n", hydro->shearViscosityToEntropyDensity);

	// Initialize CUDA kernel parameters
	initializeCUDALaunchParameters(latticeParams);
	initializeCUDAConstantParameters(latticeParams, initCondParams, hydroParams);

	// Allocate host and device memory
	size_t bytes = nElements * sizeof(PRECISION);
	allocateHostMemory(nElements);
	allocateDeviceMemory(bytes);

	//initialize cornelius for freezeout surface finding
	//see example_4d() in example_cornelius
	int dim;
	double *lattice_spacing;
	if ((nx > 1) && (ny > 1) && (nz > 1))
	{
		dim = 4;
		lattice_spacing = new double[dim];
		lattice_spacing[0] = dt;
		lattice_spacing[1] = dx;
		lattice_spacing[2] = dy;
		lattice_spacing[3] = dz;
	}
	else if ((nx > 1) && (ny > 1) && (nz < 2))
	{
		dim = 3;
		lattice_spacing = new double[dim];
		lattice_spacing[0] = dt;
		lattice_spacing[1] = dx;
		lattice_spacing[2] = dy;
	}
	else
	{
		printf("simulation is not in 3+1D or 2+1D; freezeout finder will not work!\n");
	}

	Cornelius cor;
	cor.init(dim, freezeoutEnergyDensity, lattice_spacing);

	double ****energy_density_evoution;
	energy_density_evoution = calloc4dArray(energy_density_evoution, FOFREQ+1, nx, ny, nz);

	//make an array to store all the hydrodynamic variables for FOFREQ time steps
	//to be written to file once the freezeout surface is determined by the critical energy density
	int n_hydro_vars = 16; //u0, u1, u2, u3, e, pi00, pi01, pi02, pi03, pi11, pi12, pi13, pi22, pi23, pi33, Pi, the temperature and pressure are calclated with EoS
	double *****hydrodynamic_evoution;
	hydrodynamic_evoution = calloc5dArray(hydrodynamic_evoution, n_hydro_vars, FOFREQ+1, nx, ny, nz);

	//for 3+1D simulations
	double ****hyperCube4D;
	hyperCube4D = calloc4dArray(hyperCube4D, 2, 2, 2, 2);
	//for 2+1D simulations
	double ***hyperCube3D;
	hyperCube3D = calloc3dArray(hyperCube3D, 2, 2, 2);

	//open the freezeout surface file
	ofstream freezeoutSurfaceFile;
	if (FOFORMAT == 0) freezeoutSurfaceFile.open("output/surface.dat", ios::out);
	else freezeoutSurfaceFile.open("output/surface.dat", ios::out | ios::binary);

	/************************************************************************************\
	* Fluid dynamic initialization
	/************************************************************************************/
	double t = t0;
	// generate initial conditions
	setInitialConditions(latticeParams, initCondParams, hydroParams, rootDirectory);
	// Calculate conserved quantities
	setConservedVariables(t, latticeParams);
	// copy conserved/inferred variables to GPU memory
	copyHostToDeviceMemory(bytes);
	// impose boundary conditions with ghost cells
	setGhostCells(d_q,d_e,d_p,d_u);
	//#ifndef IDEAL
	checkValidity(t, d_validityDomain, d_q, d_e, d_p, d_u, d_up);
	//#endif
	/************************************************************************************\
	* Evolve the system in time
	/************************************************************************************/
	int ictr = (nx % 2 == 0) ? ncx/2 : (ncx-1)/2;
	int jctr = (ny % 2 == 0) ? ncy/2 : (ncy-1)/2;
	int kctr = (nz % 2 == 0) ? ncz/2 : (ncz-1)/2;
	int sctr = columnMajorLinearIndex(ictr, jctr, kctr, ncx, ncy);

	Stopwatch sw;
	double totalTime = 0;
	int nsteps = 0;

	int accumulator1 = 0;
	int accumulator2 = 0;

	for (int n = 1; n <= nt+1; ++n) {
		// copy variables back to host and write to disk
		if ((n-1) % FREQ == 0)
		{
			copyDeviceToHostMemory(bytes);
			printf("n = %d:%d (t = %.3f),\t (e, p) = (%.3f, %.3f) [GeV/fm^3],\t (T = %.3f [GeV]),\t",
			n - 1, nt, t, e[sctr]*hbarc, p[sctr]*hbarc, effectiveTemperature(e[sctr])*hbarc);
			outputDynamicalQuantities(t, outputDir, latticeParams);
			// end hydrodynamic simulation if the temperature is below the freezeout temperature
			//if(e[sctr] < freezeoutEnergyDensity) {
			//printf("\nReached freezeout temperature at the center.\n");
			//break;
			//}
		}

		//************************************************************************************\
		// Freeze-out finder (Derek)
		// the freezeout surface file is written in the format which can
		// be read by iS3D : https://github.com/derekeverett/iS3D
		//************************************************************************************/
		//append the energy density and all hydro variables to storage arrays
		int nFO = n % FOFREQ;
		//need all hydro info on host for FO surface finding
		//make sure this isn't being copied TWICE when (n-1) % FREQ == 0
		//check how long this copy takes - if it's a significant bottleneck?
		if ( (n-1) % FREQ != 0) copyDeviceToHostMemory(bytes);

		if(nFO == 0) //swap in the old values so that freezeout volume elements have overlap between calls to finder
		{
			swapAndSetHydroVariables(energy_density_evoution, hydrodynamic_evoution, q, e, u, nx, ny, nz, FOFREQ);
		}
		else //update the values of the rest of the array with current time step
		{
			setHydroVariables(energy_density_evoution, hydrodynamic_evoution, q, e, u, nx, ny, nz, FOFREQ, n);
		}

		//the n=1 values are written to the it = 2 index of array, so don't start until here
		int start;
		if (n <= FOFREQ) start = 2;
		else start = 0;
		if (nFO == FOFREQ - 1) //call the freezeout finder should this be put before the values are set?
		{

			//besides writing centroid and normal to file, write all the hydro variables
			int dimZ;
			if (dim == 4) dimZ = nz-1; //enter the loop over iz, and avoid problems at boundary
			else if (dim == 3) dimZ = 1; //we need to enter the 'loop' over iz rather than skipping it
			for (int it = start; it < FOFREQ; it++) //note* avoiding boundary problems (reading outside array)
			{
				for (int ix = 0; ix < nx-1; ix++)
				{
					for (int iy = 0; iy < ny-1; iy++)
					{
						for (int iz = 0; iz < dimZ; iz++)
						{
							//write the values of energy density to all corners of the hyperCube
							if (dim == 4) writeEnergyDensityToHypercube4D(hyperCube4D, energy_density_evoution, it, ix, iy, iz);
							else if (dim == 3) writeEnergyDensityToHypercube3D(hyperCube3D, energy_density_evoution, it, ix, iy);

							//use cornelius to find the centroid and normal vector of each hyperCube
							if (dim == 4) cor.find_surface_4d(hyperCube4D);
							else if (dim == 3) cor.find_surface_3d(hyperCube3D);
							//write centroid and normal of each surface element to file
							for (int i = 0; i < cor.get_Nelements(); i++)
							{
								double temp = 0.0; //temporary variable
								//first write the position of the centroid of surface element
								double cell_tau = t0 + ((double)(n - FOFREQ + it)) * dt; //check if this is the correct time!
								double cell_x = (double)ix * dx  - (((double)(nx-1)) / 2.0 * dx);
								double cell_y = (double)iy * dy  - (((double)(ny-1)) / 2.0 * dy);
								double cell_z = (double)iz * dz  - (((double)(nz-1)) / 2.0 * dz);

								double tau_frac = cor.get_centroid_elem(i,0) / lattice_spacing[0];
								double x_frac = cor.get_centroid_elem(i,1) / lattice_spacing[1];
								double y_frac = cor.get_centroid_elem(i,2) / lattice_spacing[2];
								double z_frac;
								if (dim == 4) z_frac = cor.get_centroid_elem(i,3) / lattice_spacing[3];
								else z_frac = 0.0;

								if (FOFORMAT == 0) //write ASCII file
								{
									//first write the contravariant position vector
									if (FOTEST) {freezeoutSurfaceFile << cell_tau << " ";}
									else {freezeoutSurfaceFile << cor.get_centroid_elem(i,0) + cell_tau << " ";}
									freezeoutSurfaceFile << cor.get_centroid_elem(i,1) + cell_x << " ";
									freezeoutSurfaceFile << cor.get_centroid_elem(i,2) + cell_y << " ";
									if (dim == 4) freezeoutSurfaceFile << cor.get_centroid_elem(i,3) + cell_z << " ";
									else freezeoutSurfaceFile << cell_z << " ";
									//then the (covariant?) surface normal element; check jacobian factors of tau for milne coordinates!
                  //acording to cornelius user guide, corenelius returns covariant components of normal vector without jacobian factors
									freezeoutSurfaceFile << t * cor.get_normal_elem(i,0) << " ";
									freezeoutSurfaceFile << t * cor.get_normal_elem(i,1) << " ";
									freezeoutSurfaceFile << t * cor.get_normal_elem(i,2) << " ";
									if (dim == 4) freezeoutSurfaceFile << t * cor.get_normal_elem(i,3) << " ";
									else freezeoutSurfaceFile << 0.0 << " ";
									//write all the necessary hydro dynamic variables by first performing linear interpolation from values at
									//corners of hypercube

									if (dim == 4) // for 3+1D
									{
										//first write the contravariant flow velocity
										for (int ivar = 0; ivar < dim; ivar++)
										{
											temp = interpolateVariable4D(hydrodynamic_evoution, ivar, it, ix, iy, iz, tau_frac, x_frac, y_frac, z_frac);
											freezeoutSurfaceFile << temp << " ";
										}
										//write the energy density
										temp = interpolateVariable4D(hydrodynamic_evoution, 4, it, ix, iy, iz, tau_frac, x_frac, y_frac, z_frac);
										freezeoutSurfaceFile << temp << " "; //note : iSpectra reads in file in fm^x units e.g. energy density should be written in fm^-4
										//the temperature !this needs to be checked
										freezeoutSurfaceFile << effectiveTemperature(temp) << " ";
										//the thermal pressure
										freezeoutSurfaceFile << equilibriumPressure(temp) << " ";
										//write ten components of pi_(mu,nu) shear viscous tensor
										for (int ivar = 5; ivar < 15; ivar++)
										{
											temp = interpolateVariable4D(hydrodynamic_evoution, ivar, it, ix, iy, iz, tau_frac, x_frac, y_frac, z_frac);
											freezeoutSurfaceFile << temp << " ";
										}
										//write the bulk pressure Pi, and start a new line
										temp = interpolateVariable4D(hydrodynamic_evoution, 15, it, ix, iy, iz, tau_frac, x_frac, y_frac, z_frac);
										freezeoutSurfaceFile << temp << endl;
									}

									else //for 2+1D
									{
										//first write the contravariant flow velocity
										for (int ivar = 0; ivar < 4; ivar++)
										{
											temp = interpolateVariable3D(hydrodynamic_evoution, ivar, it, ix, iy, tau_frac, x_frac, y_frac);
											freezeoutSurfaceFile << temp << " ";
										}
										//write the energy density
										temp = interpolateVariable3D(hydrodynamic_evoution, 4, it, ix, iy, tau_frac, x_frac, y_frac);
										freezeoutSurfaceFile << temp << " "; //note units of fm^-4 appropriate for iSpectra reading
										//the temperature !this needs to be checked
										freezeoutSurfaceFile << effectiveTemperature(temp) << " ";
										//the thermal pressure
										freezeoutSurfaceFile << equilibriumPressure(temp) << " ";
										//write ten components of pi_(mu,nu) shear viscous tensor
										for (int ivar = 5; ivar < 15; ivar++)
										{
											temp = interpolateVariable3D(hydrodynamic_evoution, ivar, it, ix, iy, tau_frac, x_frac, y_frac);
											freezeoutSurfaceFile << temp << " ";
										}
										//write the bulk pressure Pi, and start a new line
										temp = interpolateVariable3D(hydrodynamic_evoution, 15, it, ix, iy, tau_frac, x_frac, y_frac);
										freezeoutSurfaceFile << temp << endl;
									}
								}

								else //write in binary
								{
									//first write the contravariant position vector
									double pos_tau = cor.get_centroid_elem(i,0) + cell_tau;
									double pos_x = cor.get_centroid_elem(i,1) + cell_x;
									double pos_y = cor.get_centroid_elem(i,2) + cell_y;
									double pos_z = 0.0;
									if (dim == 4) pos_z = cor.get_centroid_elem(i,3) + cell_z;
									freezeoutSurfaceFile.write( (char*)&pos_tau, sizeof(double));
									freezeoutSurfaceFile.write( (char*)&pos_x, sizeof(double));
									freezeoutSurfaceFile.write( (char*)&pos_y, sizeof(double));
									if (dim == 4) freezeoutSurfaceFile.write( (char*)&pos_z, sizeof(double));

									//then the (covariant?) surface normal element; check jacobian factors of tau for milne coordinates!
                  //acording to cornelius user guide, corenelius returns covariant components of normal vector without jacobian factors
									double normal_tau = t * cor.get_normal_elem(i,0);
									double normal_x = t * cor.get_normal_elem(i,1);
									double normal_y = t * cor.get_normal_elem(i,2);
									double normal_z = 0.0;
									if (dim == 4) normal_z = t * cor.get_normal_elem(i,3);
									freezeoutSurfaceFile.write( (char*)&normal_tau, sizeof(double));
									freezeoutSurfaceFile.write( (char*)&normal_x, sizeof(double));
									freezeoutSurfaceFile.write( (char*)&normal_y, sizeof(double));
									if (dim == 4)freezeoutSurfaceFile.write( (char*)&normal_z, sizeof(double));

									//write all the necessary hydro dynamic variables by first performing linear interpolation from values at
									//corners of hypercube

									if (dim == 4) // for 3+1D
									{
										//first write the contravariant flow velocity
										for (int ivar = 0; ivar < dim; ivar++)
										{
											temp = interpolateVariable4D(hydrodynamic_evoution, ivar, it, ix, iy, iz, tau_frac, x_frac, y_frac, z_frac);
											freezeoutSurfaceFile.write( (char*)&temp, sizeof(double));
										}
										//write the energy density
										temp = interpolateVariable4D(hydrodynamic_evoution, 4, it, ix, iy, iz, tau_frac, x_frac, y_frac, z_frac);
										freezeoutSurfaceFile.write( (char*)&temp, sizeof(double)); //note : iSpectra reads in file in fm^x units e.g. energy density should be written in fm^-4
										//the temperature !this needs to be checked
										double fo_temperature = effectiveTemperature(temp);
										freezeoutSurfaceFile.write( (char*)&fo_temperature, sizeof(double));
										//the thermal pressure
										double fo_pressure = equilibriumPressure(temp);
										freezeoutSurfaceFile.write( (char*)&fo_pressure, sizeof(double));
										//write ten contravariant components of pi_(mu,nu) shear viscous tensor
										for (int ivar = 5; ivar < 15; ivar++)
										{
											temp = interpolateVariable4D(hydrodynamic_evoution, ivar, it, ix, iy, iz, tau_frac, x_frac, y_frac, z_frac);
											freezeoutSurfaceFile.write( (char*)&temp, sizeof(double));
										}
										//write the bulk pressure Pi, and start a new line
										temp = interpolateVariable4D(hydrodynamic_evoution, 15, it, ix, iy, iz, tau_frac, x_frac, y_frac, z_frac);
										freezeoutSurfaceFile.write( (char*)&temp, sizeof(double));
									}

									else //for 2+1D
									{
										//first write the contravariant flow velocity
										for (int ivar = 0; ivar < 4; ivar++)
										{
											temp = interpolateVariable3D(hydrodynamic_evoution, ivar, it, ix, iy, tau_frac, x_frac, y_frac);
											freezeoutSurfaceFile.write( (char*)&temp, sizeof(double));
										}
										//write the energy density
										temp = interpolateVariable3D(hydrodynamic_evoution, 4, it, ix, iy, tau_frac, x_frac, y_frac);
										freezeoutSurfaceFile.write( (char*)&temp, sizeof(double)); //note units of fm^-4 appropriate for iSpectra reading
										//the temperature !this needs to be checked
										double fo_temperature = effectiveTemperature(temp);
										freezeoutSurfaceFile.write( (char*)&fo_temperature, sizeof(double));
										//the thermal pressure
										double fo_pressure = equilibriumPressure(temp);
										freezeoutSurfaceFile.write( (char*)&fo_pressure, sizeof(double));
										//write ten components of pi_(mu,nu) shear viscous tensor
										for (int ivar = 5; ivar < 15; ivar++)
										{
											temp = interpolateVariable3D(hydrodynamic_evoution, ivar, it, ix, iy, tau_frac, x_frac, y_frac);
											freezeoutSurfaceFile.write( (char*)&temp, sizeof(double));
										}
										//write the bulk pressure Pi, and start a new line
										temp = interpolateVariable3D(hydrodynamic_evoution, 15, it, ix, iy, tau_frac, x_frac, y_frac);
										freezeoutSurfaceFile.write( (char*)&temp, sizeof(double));
									}
								}
							}
						}
					}
				}
			}
		}

//if all cells are below freezeout temperature end hydro
accumulator1 = 0;
for (int ix = 2; ix < nx+2; ix++)
{
	for (int iy = 2; iy < ny+2; iy++)
	{
		for (int iz = 2; iz < nz+2; iz++)
		{
			int s = columnMajorLinearIndex(ix, iy, iz, nx+4, ny+4);
			if (e[s] > freezeoutEnergyDensity) accumulator1 = accumulator1 + 1;
		}
	}
}
if (accumulator1 == 0) accumulator2 += 1;
if (accumulator2 >= FOFREQ+1) //only break once freezeout finder has had a chance to search/write to file
{
	printf("\nAll cells have dropped below freezeout energy density\n");
	break;
}
sw.tic();
twoStepRungeKutta(t, dt, d_q, d_Q);
sw.toc();
float elapsedTime = sw.elapsedTime();
if ((n-1) % FREQ == 0) printf("(Elapsed time/step: %.3f ms)\n", elapsedTime);
totalTime+=elapsedTime;
++nsteps;

setCurrentConservedVariables();

t = t0 + n * dt;
}
printf("Average time/step: %.3f ms\n",totalTime/((double)nsteps));

/************************************************************************************\
* Deallocate host and device memory
/************************************************************************************/
freeHostMemory();
freeDeviceMemory();
cudaDeviceReset();
}
