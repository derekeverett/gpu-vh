#include <stdlib.h>
#include <stdio.h> // for printf

#include <cuda.h>
#include <cuda_runtime.h>

#include "../include/LatticeParameters.h"
#include "../include/DynamicalVariables.cuh"
#include "../include/CudaConfiguration.cuh"
#include "../include/RegulateDissipativeCurrents.cuh"

//#define REGULATE_BULK

#ifndef IDEAL
__global__
void regulateDissipativeCurrents(PRECISION t,
CONSERVED_VARIABLES * const __restrict__ currentVars,
const PRECISION * const __restrict__ e, const PRECISION * const __restrict__ p,
const FLUID_VELOCITY * const __restrict__ u,
VALIDITY_DOMAIN * const __restrict__ validityDomain
) {
	unsigned int threadID = blockDim.x * blockIdx.x + threadIdx.x;
	if (threadID < d_nElements) {
		unsigned int k = threadID / (d_nx * d_ny) + N_GHOST_CELLS_M;
		unsigned int j = (threadID % (d_nx * d_ny)) / d_nx + N_GHOST_CELLS_M;
		unsigned int i = threadID % d_nx + N_GHOST_CELLS_M;
		unsigned int s = columnMajorLinearIndex(i, j, k, d_ncx, d_ncy);
		#ifdef PIMUNU
		PRECISION pitt = currentVars->pitt[s];
		PRECISION pitx = currentVars->pitx[s];
		PRECISION pity = currentVars->pity[s];
		PRECISION pitn = currentVars->pitn[s];
		PRECISION pixx = currentVars->pixx[s];
		PRECISION pixy = currentVars->pixy[s];
		PRECISION pixn = currentVars->pixn[s];
		PRECISION piyy = currentVars->piyy[s];
		PRECISION piyn = currentVars->piyn[s];
		PRECISION pinn = currentVars->pinn[s];
		#else
		PRECISION pitt = 0.0;
		PRECISION pitx = 0.0;
		PRECISION pity = 0.0;
		PRECISION pitn = 0.0;
		PRECISION pixx = 0.0;
		PRECISION pixy = 0.0;
		PRECISION pixn = 0.0;
		PRECISION piyy = 0.0;
		PRECISION piyn = 0.0;
		PRECISION pinn = 0.0;
		#endif
		#ifdef Pi
		PRECISION Pi = currentVars->Pi[s];
		#else
		PRECISION Pi = 0;
		#endif

		PRECISION ut = u->ut[s];
		PRECISION ux = u->ux[s];
		PRECISION uy = u->uy[s];
		PRECISION un = u->un[s];

		PRECISION e_s = e[s];
		PRECISION p_s = p[s];

		//xi0 controls regulation of tracelessness condition, according to iEBE best chosen to be 0.1
		PRECISION xi0 = 0.1;
		//rhomax best chosen to be in ( 1.0, 10.0) according to iEBE paper ; smaller rhomax -> stronger shear regulation
		//rhomax controls inv reynolds number
		PRECISION rhomax = 1.0;

		PRECISION t2 = t*t;

		PRECISION pipi = pitt*pitt-2*pitx*pitx-2*pity*pity+pixx*pixx+2*pixy*pixy+piyy*piyy-2*pitn*pitn*t2+2*pixn*pixn*t2+2*piyn*piyn*t2+pinn*pinn*t2*t2;
		//PRECISION spipi = sqrtf(fabsf(pipi+3*Pi*Pi));
		PRECISION spipi = sqrtf(fabsf(pipi));
		PRECISION pimumu = pitt - pixx - piyy - pinn*t*t;
		PRECISION piu0 = -pitn*t2*un + pitt*ut - pitx*ux - pity*uy;
		PRECISION piu1 = -pixn*t2*un + pitx*ut - pixx*ux - pixy*uy;
		PRECISION piu2 = -piyn*t2*un + pity*ut - pixy*ux - piyy*uy;
		PRECISION piu3 = -pinn*t2*un + pitn*ut - pixn*ux - piyn*uy;

		PRECISION a1 = fdividef(spipi, rhomax)*rsqrtf(e_s*e_s+3*p_s*p_s);
		PRECISION den = xi0*rhomax*spipi;
///*
		PRECISION a2 = fdividef(pimumu, den);
		PRECISION a3 = fdividef(piu0, den);
		PRECISION a4 = fdividef(piu1, den);
		PRECISION a5 = fdividef(piu2, den);
		PRECISION a6 = fdividef(piu3, den);
//*/
/*
		PRECISION a2 = fabsf(fdividef(pimumu, den));
		PRECISION a3 = fabsf(fdividef(piu0, den));
		PRECISION a4 = fabsf(fdividef(piu1, den));
		PRECISION a5 = fabsf(fdividef(piu2, den));
		PRECISION a6 = fabsf(fdividef(piu3, den));
//*/
		PRECISION rho = fmaxf(a1,fmaxf(a2,fmaxf(a3,fmaxf(a4,fmaxf(a5,a6)))));

		PRECISION fac = fdividef(tanhf(rho), rho);
		if(fabsf(rho)<1.e-7) fac = 1;

		//regulate the shear stress
		#ifdef PIMUNU
		currentVars->pitt[s] *= fac;
		currentVars->pitx[s] *= fac;
		currentVars->pity[s] *= fac;
		currentVars->pitn[s] *= fac;
		currentVars->pixx[s] *= fac;
		currentVars->pixy[s] *= fac;
		currentVars->pixn[s] *= fac;
		currentVars->piyy[s] *= fac;
		currentVars->piyn[s] *= fac;
		currentVars->pinn[s] *= fac;
		#endif

		//regulate the bulk pressure according to it's inverse reynolds #
		#ifdef REGULATE_BULK
		PRECISION rhoBulk = abs(Pi) / sqrtf(e_s * e_s + 3 * p_s * p_s);
		//PRECISION rhoBulk = abs(Pi) / p_s;
		if(isnan(rhoBulk) == 1) printf("found rhoBulk Nan\n");
		PRECISION facBulk = tanh(rhoBulk) / rhoBulk;
		if(fabs(rhoBulk) < 1.e-7) facBulk = 1.0;
		if(isnan(facBulk) == 1) printf("found facBulk Nan\n");

		//regulate bulk pressure
		#ifdef PI
		currentVars->Pi[s] *= facBulk;
		#endif
		#endif

		validityDomain->regulations[s] = fac;
	}
}
#endif
