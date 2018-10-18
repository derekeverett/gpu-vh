/*
 * FullyDiscreteKurganovTadmorScheme.cuh
 *
 *  Created on: Oct 23, 2015
 *      Author: bazow
 */

#ifndef FULLYDISCRETEKURGANOVTADMORSCHEME_CUH_
#define FULLYDISCRETEKURGANOVTADMORSCHEME_CUH_

#include "DynamicalVariables.cuh"

__global__
void convexCombinationEulerStepKernel(
		const CONSERVED_VARIABLES * const __restrict__ q,
		CONSERVED_VARIABLES * const __restrict__ Q);

void twoStepRungeKutta(PRECISION t, PRECISION dt,
		CONSERVED_VARIABLES * __restrict__ d_q,
		CONSERVED_VARIABLES * __restrict__ d_Q, PRECISION T_reg);

#endif /* FULLYDISCRETEKURGANOVTADMORSCHEME_CUH_ */
