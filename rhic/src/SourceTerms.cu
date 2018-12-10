/*
 * SourceTerms.cu
 *
 *  Created on: Oct 22, 2015
 *      Author: bazow
 */

#include <stdio.h> // for printf
#include <math.h> // for math functions

#include <cuda.h>
#include <cuda_runtime.h>

#include "../include/SourceTerms.cuh"
#include "../include/FiniteDifference.cuh"
#include "../include/EnergyMomentumTensor.cuh"
#include "../include/DynamicalVariables.cuh"
#include "../include/CudaConfiguration.cuh"

#include "../include/EquationOfState.cuh" // for bulk terms
#include "../include/TransportCoefficients.cuh"

#include "../include/FluxLimiter.cuh"

//#define USE_CARTESIAN_COORDINATES

#define USE_APPROX_DERIVATIVE //use approximate derivative with minmod flux limiter for derivatives of flow and pressure

__device__
void setPimunuSourceTerms(PRECISION * const __restrict__ pimunuRHS,
PRECISION t, PRECISION e, PRECISION p,
PRECISION ut, PRECISION ux, PRECISION uy, PRECISION un, PRECISION utp,
PRECISION uxp, PRECISION uyp, PRECISION unp,
PRECISION pitt, PRECISION pitx, PRECISION pity,
PRECISION pitn, PRECISION pixx, PRECISION pixy, PRECISION pixn,
PRECISION piyy,
PRECISION piyn, PRECISION pinn, PRECISION Pi,
PRECISION dxut, PRECISION dyut, PRECISION dnut, PRECISION dxux,
PRECISION dyux, PRECISION dnux,
PRECISION dxuy, PRECISION dyuy, PRECISION dnuy, PRECISION dxun,
PRECISION dyun, PRECISION dnun, PRECISION dkvk) {
	
	/*********************************************************\
	 * Temperature dependent shear transport coefficients
	 /*********************************************************/
	PRECISION T = effectiveTemperature(e);
	PRECISION taupiInv = 0.2f * fdividef(T, d_etabar);
	PRECISION beta_pi = 0.2f * (e + p);

	/*********************************************************\
	 * Temperature dependent bulk transport coefficients
	 /*********************************************************/
	PRECISION cs2 = speedOfSoundSquared(e);
	PRECISION a = 0.333333f - cs2;
	PRECISION a2 = a * a;
	PRECISION beta_Pi = 15.0 * a2 * (e + p);
	PRECISION lambda_Pipi = 1.6f * a;

	PRECISION zetabar = bulkViscosityToEntropyDensity(T);
	PRECISION tauPiInv = 15 * a2 * fdividef(T, zetabar);

	PRECISION ut2 = ut * ut;
	PRECISION un2 = un * un;
	PRECISION t2 = t * t;
	PRECISION t3 = t * t2;

	// time derivatives of u
	PRECISION dtut = (ut - utp) / d_dt;
	PRECISION dtux = (ux - uxp) / d_dt;
	PRECISION dtuy = (uy - uyp) / d_dt;
	PRECISION dtun = (un - unp) / d_dt;

	/*********************************************************\
	 * covariant derivatives
	 /*********************************************************/
	PRECISION Dut = ut * dtut + ux * dxut + uy * dyut + un * dnut + t * un * un;
	PRECISION dut = Dut - t * un * un;

	PRECISION dux = ut * dtux + ux * dxux + uy * dyux + un * dnux;
	PRECISION Dux = -dux;

	PRECISION duy = ut * dtuy + ux * dxuy + uy * dyuy + un * dnuy;
	PRECISION Duy = -duy;

	PRECISION dun = ut * dtun + ux * dxun + uy * dyun + un * dnun;
	PRECISION Dun = -t2 * dun - 2.0 * t * ut * un;

	/*********************************************************\
	 * expansion rate
	 /*********************************************************/
	PRECISION theta = ut / t + dtut + dxux + dyuy + dnun;

	/*********************************************************\
	 * Velocity shear stress tensor
	 /*********************************************************/
	PRECISION theta3 = theta / 3.0;
	PRECISION stt = -t * ut * un2 + (dtut - ut * dut) + (ut2 - 1.0) * theta3;
	PRECISION stx = -(t * un2 * ux) / 2.0 + (dtux - dxut) / 2.0 - (ux * dut + ut * dux) / 2.0 + ut * ux * theta3;
	PRECISION sty = -(t * un2 * uy) / 2.0 + (dtuy - dyut) / 2.0 - (uy * dut + ut * duy) / 2.0 + ut * uy * theta3;
	PRECISION stn = -un * (2.0 * ut2 + t2 * un2) / (2.0 * t) + (dtun - dnut / t2) / 2.0 - (un * dut + ut * dun) / 2.0 + ut * un * theta3;
	PRECISION sxx = -(dxux + ux * dux) + (1.0 + ux * ux) * theta3;
	PRECISION sxy = -(dxuy + dyux) / 2.0 - (uy * dux + ux * duy) / 2.0 + ux * uy * theta3;
	PRECISION sxn = -ut * ux * un / t - (dxun + dnux / t2) / 2.0 - (un * dux + ux * dun) / 2.0 + ux * un * theta3;
	PRECISION syy = -(dyuy + uy * duy) + (1.0 + uy * uy) * theta3;
	PRECISION syn = -ut * uy * un / t - (dyun + dnuy / t2) / 2.0 - (un * duy + uy * dun) / 2.0 + uy * un * theta3;
	PRECISION snn = -ut * (1.0 + 2.0 * t2 * un2) / t3 - dnun / t2 - un * dun + (1.0 / t2 + un2) * theta3;

	/*********************************************************\
	 * vorticity tensor w^\mu _\nu
	/*********************************************************/

	//Original expressions were inconsistent with DNMR and Eqns of Motion up to signs and factors of 1/2
	/*
	PRECISION wtx = (dtux + dxut) / 2 + (ux * dut - ut * dux) / 2 + t * un2 * ux / 2;
	PRECISION wty = (dtuy + dyut) / 2 + (uy * dut - ut * duy) / 2 + t * un2 * uy / 2;
	PRECISION wtn = (t2 * dtun + 2 * t * un + dnut) / 2 + (t2 * un * dut - ut * Dun) + t3 * un*un2 / 2;
	PRECISION wxy = (dyux - dxuy) / 2 + (uy * dux - ux * duy) / 2;
	PRECISION wxn = (dnux - t2 * dxun) / 2 + (t2 * un * dux - ux * Dun) / 2;
	PRECISION wyn = (dnuy - t2 * dyun) / 2 + (t2 * un * duy - uy * Dun) / 2;
	*/

	//these expressions verified by McNelis and Du
	PRECISION wtx = (-dtux  -  dxut  +  ut * dux  -  ux * dut  -  t * un2 * ux) / 2.0;
	PRECISION wty = (-dtuy  -  dyut  +  ut * duy  -  uy * dut  -  t * un2 * uy) / 2.0;
	PRECISION wtn = (-t2 * dtun  -  2.0 * t * un  - dnut  -  ut * Dun  - t2 * un * dut  -  t3 * un * un2) / 2.0;
	PRECISION wxy = (dxuy  -  dyux  +  ux * duy  -  uy * dux) / 2.0;
	PRECISION wxn = (t2 * dxun  -  dnux  -  ux * Dun  -  t2 * un * dux) / 2.0;
	PRECISION wyn = (t2 * dyun  -  dnuy  -  uy * Dun  -  t2 * un * duy) / 2.0;

	// anti-symmetric vorticity components
	PRECISION wxt = wtx;
	PRECISION wyt = wty;
	PRECISION wnt = wtn / t2;
	PRECISION wyx = -wxy;
	PRECISION wnx = -wxn / t2;
	PRECISION wny = -wyn / t2;

	/*********************************************************\
	 * I1
	 /*********************************************************/
	PRECISION I1tt = 2.0 * ut * (pitt * Dut + pitx * Dux + pity * Duy + pitn * Dun);
	PRECISION I1tx = (pitt * ux + pitx * ut) * Dut + (pitx * ux + pixx * ut) * Dux + (pity * ux + pixy * ut) * Duy + (pitn * ux + pixn * ut) * Dun;
	PRECISION I1ty = (pitt * uy + pity * ut) * Dut + (pitx * uy + pixy * ut) * Dux + (pity * uy + piyy * ut) * Duy + (pitn * uy + piyn * ut) * Dun;
	PRECISION I1tn = (pitt * un + pitn * ut) * Dut + (pitx * un + pixn * ut) * Dux + (pity * un + piyn * ut) * Duy + (pitn * un + pinn * ut) * Dun;
	PRECISION I1xx = 2.0 * ux * (pitx * Dut + pixx * Dux + pixy * Duy + pixn * Dun);
	PRECISION I1xy = (pitx * uy + pity * ux) * Dut + (pixx * uy + pixy * ux) * Dux + (pixy * uy + piyy * ux) * Duy + (pixn * uy + piyn * ux) * Dun;
	PRECISION I1xn = (pitx * un + pitn * ux) * Dut + (pixx * un + pixn * ux) * Dux + (pixy * un + piyn * ux) * Duy + (pixn * un + pinn * ux) * Dun;
	PRECISION I1yy = 2.0 * uy * (pity * Dut + pixy * Dux + piyy * Duy + piyn * Dun);
	PRECISION I1yn = (pity * un + pitn * uy) * Dut + (pixy * un + pixn * uy) * Dux + (piyy * un + piyn * uy) * Duy + (piyn * un + pinn * uy) * Dun;
	PRECISION I1nn = 2.0 * un * (pitn * Dut + pixn * Dux + piyn * Duy + pinn * Dun);

	/*********************************************************\
	 * I2
	 /*********************************************************/
	PRECISION I2tt = theta * pitt;
	PRECISION I2tx = theta * pitx;
	PRECISION I2ty = theta * pity;
	PRECISION I2tn = theta * pitn;
	PRECISION I2xx = theta * pixx;
	PRECISION I2xy = theta * pixy;
	PRECISION I2xn = theta * pixn;
	PRECISION I2yy = theta * piyy;
	PRECISION I2yn = theta * piyn;
	PRECISION I2nn = theta * pinn;

	/*********************************************************\
	 * I3
	 /*********************************************************/
	PRECISION I3tt = 2.0 * (pitx * wtx + pity * wty + pitn * wtn);
	PRECISION I3tx = pitt * wxt + pity * wxy + pitn * wxn + pixx * wtx + pixy * wty + pixn * wtn;
	PRECISION I3ty = pitt * wyt + pitx * wyx + pitn * wyn + pixy * wtx + piyy * wty + piyn * wtn;
	PRECISION I3tn = pitt * wnt + pitx * wnx + pity * wny + pixn * wtx + piyn * wty + pinn * wtn;
	PRECISION I3xx = 2.0 * (pitx * wxt + pixy * wxy + pixn * wxn);
	PRECISION I3xy = pitx * wyt + pity * wxt + pixx * wyx + piyy * wxy + pixn * wyn + piyn * wxn;
	PRECISION I3xn = pitx * wnt + pitn * wxt + pixx * wnx + pixy * wny + piyn * wxy + pinn * wxn;
	PRECISION I3yy = 2.0 * (pity * wyt + pixy * wyx + piyn * wyn);
	PRECISION I3yn = pity * wnt + pitn * wyt + pixy * wnx + pixn * wyx + piyy * wny + pinn * wyn;
	PRECISION I3nn = 2.0 * (pitn * wnt + pixn * wnx + piyn * wny);

	/*********************************************************\
	 * I4
	 /*********************************************************/
	PRECISION ux2 = ux * ux;
	PRECISION uy2 = uy * uy;
	PRECISION ps = pitt * stt - 2.0 * pitx * stx - 2.0 * pity * sty + pixx * sxx + 2.0 * pixy * sxy + piyy * syy - 2.0 * pitn * stn * t2 + 2.0 * pixn * sxn * t2
			+ 2.0 * piyn * syn * t2 + pinn * snn * t2 * t2;
	PRECISION ps3 = ps / 3.0;
	PRECISION I4tt = (pitt * stt - pitx * stx - pity * sty - t2 * pitn * stn) - (1 - ut2) * ps3;
	PRECISION I4tx = (pitt * stx + pitx * stt) / 2.0 - (pitx * sxx + pixx * stx) / 2.0 - (pity * sxy + pixy * sty) / 2.0 - t2 * (pitn * sxn + pixn * stn) / 2.0
			+ (ut * ux) * ps3;
	PRECISION I4ty = (pitt * sty + pity * stt) / 2.0 - (pitx * sxy + pixy * stx) / 2.0 - (pity * syy + piyy * sty) / 2.0 - t2 * (pitn * syn + piyn * stn) / 2.0
			+ (ut * uy) * ps3;
	PRECISION I4tn = (pitt * stn + pitn * stt) / 2.0 - (pitx * sxn + pixn * stx) / 2.0 - (pity * syn + piyn * sty) / 2.0 - t2 * (pitn * snn + pinn * stn) / 2.0
			+ (ut * un) * ps3;
	PRECISION I4xx = (pitx * stx - pixx * sxx - pixy * sxy - t2 * pixn * sxn) + (1.0 + ux2) * ps3;
	PRECISION I4xy = (pitx * sty + pity * stx) / 2.0 - (pixx * sxy + pixy * sxx) / 2.0 - (pixy * syy + piyy * sxy) / 2.0 - t2 * (pixn * syn + piyn * sxn) / 2.0
			+ (ux * uy) * ps3;
	PRECISION I4xn = (pitx * stn + pitn * stx) / 2.0 - (pixx * sxn + pixn * sxx) / 2.0 - (pixy * syn + piyn * sxy) / 2.0 - t2 * (pixn * snn + pinn * sxn) / 2.0
			+ (ux * un) * ps3;
	PRECISION I4yy = (pity * sty - pixy * sxy - piyy * syy - t2 * piyn * syn) + (1.0 + uy2) * ps3;
	PRECISION I4yn = (pity * stn + pitn * sty) / 2.0 - (pixy * sxn + pixn * sxy) / 2.0 - (piyy * syn + piyn * syy) / 2.0 - t2 * (piyn * snn + pinn * syn) / 2.0
			+ (uy * un) * ps3;
	PRECISION I4nn = (pitn * stn - pixn * sxn - piyn * syn - t2 * pinn * snn) + (1.0 / t2 + un2) * ps3;

	/*********************************************************\
	 * I
	 /*********************************************************/

	PRECISION Itt = I1tt + delta_pipi * I2tt - I3tt + tau_pipi * I4tt - lambda_piPi * Pi * stt;
	PRECISION Itx = I1tx + delta_pipi * I2tx - I3tx + tau_pipi * I4tx - lambda_piPi * Pi * stx;
	PRECISION Ity = I1ty + delta_pipi * I2ty - I3ty + tau_pipi * I4ty - lambda_piPi * Pi * sty;
	PRECISION Itn = I1tn + delta_pipi * I2tn - I3tn + tau_pipi * I4tn - lambda_piPi * Pi * stn;
	PRECISION Ixx = I1xx + delta_pipi * I2xx - I3xx + tau_pipi * I4xx - lambda_piPi * Pi * sxx;
	PRECISION Ixy = I1xy + delta_pipi * I2xy - I3xy + tau_pipi * I4xy - lambda_piPi * Pi * sxy;
	PRECISION Ixn = I1xn + delta_pipi * I2xn - I3xn + tau_pipi * I4xn - lambda_piPi * Pi * sxn;
	PRECISION Iyy = I1yy + delta_pipi * I2yy - I3yy + tau_pipi * I4yy - lambda_piPi * Pi * syy;
	PRECISION Iyn = I1yn + delta_pipi * I2yn - I3yn + tau_pipi * I4yn - lambda_piPi * Pi * syn;
	PRECISION Inn = I1nn + delta_pipi * I2nn - I3nn + tau_pipi * I4nn - lambda_piPi * Pi * snn;

	/*********************************************************\
	 * shear stress tensor source terms, i.e. terms on RHS
	 /*********************************************************/

	PRECISION dpitt = 2.0 * beta_pi * stt - pitt * taupiInv - Itt - 2.0 * un * t * pitn;
	PRECISION dpitx = 2.0 * beta_pi * stx - pitx * taupiInv - Itx - un * t * pixn;
	PRECISION dpity = 2.0 * beta_pi * sty - pity * taupiInv - Ity - un * t * piyn;
	PRECISION dpitn = 2.0 * beta_pi * stn - pitn * taupiInv - Itn - un * t * pinn - (ut * pitn + un * pitt) / t;
	PRECISION dpixx = 2.0 * beta_pi * sxx - pixx * taupiInv - Ixx;
	PRECISION dpixy = 2.0 * beta_pi * sxy - pixy * taupiInv - Ixy;
	PRECISION dpixn = 2.0 * beta_pi * sxn - pixn * taupiInv - Ixn - (ut * pixn + un * pitx) / t;
	PRECISION dpiyy = 2.0 * beta_pi * syy - piyy * taupiInv - Iyy;
	PRECISION dpiyn = 2.0 * beta_pi * syn - piyn * taupiInv - Iyn - (ut * piyn + un * pity) / t;
	PRECISION dpinn = 2.0 * beta_pi * snn - pinn * taupiInv - Inn - 2.0 * (ut * pinn + un * pitn) / t;

	/*********************************************************\
	 * bulk viscous pressure source terms, i.e. terms on RHS
	 /*********************************************************/
	PRECISION dPi = -beta_Pi * theta - Pi * tauPiInv - delta_PiPi * Pi * theta + lambda_Pipi * ps;

	/*********************************************************\
	 * time derivative of the dissipative quantities
	 /*********************************************************/
	pimunuRHS[0] = dpitt / ut + pitt * dkvk;
	pimunuRHS[1] = dpitx / ut + pitx * dkvk;
	pimunuRHS[2] = dpity / ut + pity * dkvk;
	pimunuRHS[3] = dpitn / ut + pitn * dkvk;
	pimunuRHS[4] = dpixx / ut + pixx * dkvk;
	pimunuRHS[5] = dpixy / ut + pixy * dkvk;
	pimunuRHS[6] = dpixn / ut + pixn * dkvk;
	pimunuRHS[7] = dpiyy / ut + piyy * dkvk;
	pimunuRHS[8] = dpiyn / ut + piyn * dkvk;
	pimunuRHS[9] = dpinn / ut + pinn * dkvk;
#ifdef PI
	pimunuRHS[10] = dPi / ut + Pi * dkvk;
#endif
}

/***************************************************************************************************************************************************/
__device__
void loadSourceTerms(const PRECISION * const __restrict__ I, const PRECISION * const __restrict__ J, const PRECISION * const __restrict__ K,
		const PRECISION * const __restrict__ Q,
		PRECISION * const __restrict__ S, const FLUID_VELOCITY * const __restrict__ u,
		PRECISION utp, PRECISION uxp, PRECISION uyp, PRECISION unp,
		PRECISION t, PRECISION e, const PRECISION * const __restrict__ pvec, int s) {
	//=========================================================
	// conserved variables
	//=========================================================
	PRECISION ttt = Q[0];
	PRECISION ttx = Q[1];
	PRECISION tty = Q[2];
	PRECISION ttn = Q[3];
#ifdef PIMUNU
	PRECISION pitt = Q[4];
	PRECISION pitx = Q[5];
	PRECISION pity = Q[6];
	PRECISION pitn = Q[7];
	PRECISION pixx = Q[8];
	PRECISION pixy = Q[9];
	PRECISION pixn = Q[10];
	PRECISION piyy = Q[11];
	PRECISION piyn = Q[12];
	PRECISION pinn = Q[13];
#else
	PRECISION pitt = 0;
	PRECISION pitx = 0;
	PRECISION pity = 0;
	PRECISION pitn = 0;
	PRECISION pixx = 0;
	PRECISION pixy = 0;
	PRECISION pixn = 0;
	PRECISION piyy = 0;
	PRECISION piyn = 0;
	PRECISION pinn = 0;
#endif
#ifdef PI
	PRECISION Pi = Q[14];
#else
	PRECISION Pi = 0;
#endif
	//=========================================================
	// primary variables
	//=========================================================
	PRECISION *utvec = u->ut;
	PRECISION *uxvec = u->ux;
	PRECISION *uyvec = u->uy;
	PRECISION *unvec = u->un;

	PRECISION p = pvec[s];
	PRECISION ut = utvec[s];
	PRECISION ux = uxvec[s];
	PRECISION uy = uyvec[s];
	PRECISION un = unvec[s];

	//=========================================================
	// spatial derivatives of primary variables
	//=========================================================
	PRECISION facX = 1 / d_dx / 2;
	PRECISION facY = 1 / d_dy / 2;
	PRECISION facZ = 1 / d_dz / 2;
	// dx of u^{\mu} components
	PRECISION dxut = (*(utvec + s + 1) - *(utvec + s - 1)) * facX;
	PRECISION dxux = (*(uxvec + s + 1) - *(uxvec + s - 1)) * facX;
	PRECISION dxuy = (*(uyvec + s + 1) - *(uyvec + s - 1)) * facX;
	PRECISION dxun = (*(unvec + s + 1) - *(unvec + s - 1)) * facX;
	// dy of u^{\mu} components
	PRECISION dyut = (*(utvec + s + d_ncx) - *(utvec + s - d_ncx)) * facY;
	PRECISION dyux = (*(uxvec + s + d_ncx) - *(uxvec + s - d_ncx)) * facY;
	PRECISION dyuy = (*(uyvec + s + d_ncx) - *(uyvec + s - d_ncx)) * facY;
	PRECISION dyun = (*(unvec + s + d_ncx) - *(unvec + s - d_ncx)) * facY;
	// dn of u^{\mu} components
	int stride = d_ncx * d_ncy;
	PRECISION dnut = (*(utvec + s + stride) - *(utvec + s - stride)) * facZ;
	PRECISION dnux = (*(uxvec + s + stride) - *(uxvec + s - stride)) * facZ;
	PRECISION dnuy = (*(uyvec + s + stride) - *(uyvec + s - stride)) * facZ;
	PRECISION dnun = (*(unvec + s + stride) - *(unvec + s - stride)) * facZ;
	// pressure
	PRECISION dxp = (*(pvec + s + 1) - *(pvec + s - 1)) * facX;
	PRECISION dyp = (*(pvec + s + d_ncx) - *(pvec + s - d_ncx)) * facY;
	PRECISION dnp = (*(pvec + s + stride) - *(pvec + s - stride)) * facZ;

	//=========================================================
	// spatial derivatives of the conserved variables \pi^{\mu\nu}
	//=========================================================
	int ptr = 20; // 5 * n (with n = 4 corresponding to pitt)
	PRECISION dxpitt = (*(I + ptr + 3) - *(I + ptr + 1)) * facX;
	ptr += 5;
	PRECISION dxpitx = (*(I + ptr + 3) - *(I + ptr + 1)) * facX;
	ptr += 5;
	PRECISION dxpity = (*(I + ptr + 3) - *(I + ptr + 1)) * facX;
	ptr += 5;
	PRECISION dxpitn = (*(I + ptr + 3) - *(I + ptr + 1)) * facX;
	ptr += 5;
	PRECISION dxpixx = (*(I + ptr + 3) - *(I + ptr + 1)) * facX;
	ptr += 5;
	PRECISION dxpixy = (*(I + ptr + 3) - *(I + ptr + 1)) * facX;
	ptr += 5;
	PRECISION dxpixn = (*(I + ptr + 3) - *(I + ptr + 1)) * facX;
	ptr += 20;
	PRECISION dxPi = (*(I + ptr + 3) - *(I + ptr + 1)) * facX;
	// Y
	ptr = 20; // 5 * n (with n = 4 corresponding to pitt)
	PRECISION dypitt = (*(J + ptr + 3) - *(J + ptr + 1)) * facY;
	ptr += 5;
	PRECISION dypitx = (*(J + ptr + 3) - *(J + ptr + 1)) * facY;
	ptr += 5;
	PRECISION dypity = (*(J + ptr + 3) - *(J + ptr + 1)) * facY;
	ptr += 5;
	PRECISION dypitn = (*(J + ptr + 3) - *(J + ptr + 1)) * facY;
	ptr += 10;
	PRECISION dypixy = (*(J + ptr + 3) - *(J + ptr + 1)) * facY;
	ptr += 10;
	PRECISION dypiyy = (*(J + ptr + 3) - *(J + ptr + 1)) * facY;
	ptr += 5;
	PRECISION dypiyn = (*(J + ptr + 3) - *(J + ptr + 1)) * facY;
	ptr += 10;
	PRECISION dyPi = (*(J + ptr + 3) - *(J + ptr + 1)) * facY;
	// Z
	ptr = 20; // 5 * n (with n = 4 corresponding to pitt)
	PRECISION dnpitt = (*(K + ptr + 3) - *(K + ptr + 1)) * facZ;
	ptr += 5;
	PRECISION dnpitx = (*(K + ptr + 3) - *(K + ptr + 1)) * facZ;
	ptr += 5;
	PRECISION dnpity = (*(K + ptr + 3) - *(K + ptr + 1)) * facZ;
	ptr += 5;
	PRECISION dnpitn = (*(K + ptr + 3) - *(K + ptr + 1)) * facZ;
	ptr += 15;
	PRECISION dnpixn = (*(K + ptr + 3) - *(K + ptr + 1)) * facZ;
	ptr += 10;
	PRECISION dnpiyn = (*(K + ptr + 3) - *(K + ptr + 1)) * facZ;
	ptr += 5;
	PRECISION dnpinn = (*(K + ptr + 3) - *(K + ptr + 1)) * facZ;
	ptr += 5;
	PRECISION dnPi = (*(K + ptr + 3) - *(K + ptr + 1)) * facZ;

	//=========================================================
	// T^{\mu\nu} source terms
	//=========================================================
	PRECISION tnn = Tnn(e, p + Pi, un, pinn, t);
	PRECISION vx = ux / ut;
	PRECISION vy = uy / ut;
	PRECISION vn = un / ut;
	PRECISION dxvx = (dxux - vx * dxut) / ut;
	PRECISION dyvy = (dyuy - vy * dyut) / ut;
	PRECISION dnvn = (dnun - vn * dnut) / ut;
	PRECISION dkvk = dxvx + dyvy + dnvn;
	S[0] = -(ttt / t + t * tnn) + dkvk * (pitt - p - Pi) - vx * dxp - vy * dyp - vn * dnp;
	S[1] = -ttx / t - dxp + dkvk * pitx;
	S[2] = -tty / t - dyp + dkvk * pity;
	S[3] = -3 * ttn / t - dnp / powf(t, 2.0f) + dkvk * pitn;
#ifdef USE_CARTESIAN_COORDINATES
	S[0] = dkvk*(pitt-p-Pi) - vx*dxp - vy*dyp - vn*dnp;
	S[1] = -dxp + dkvk*pitx;
	S[2] = -dyp + dkvk*pity;
	S[3] = -dnp + dkvk*pitn;
#endif

	//X
#ifndef PI
	S[0] += dxpitt*vx - dxpitx + dypitt*vy - dypity + dnpitt*vn - dnpitn;
	S[1] += dxpitx*vx - dxpixx + dypitx*vy - dypixy + dnpitx*vn - dnpixn;
	S[2] += dxpity*vx - dxpixy + dypity*vy - dypiyy + dnpity*vn - dnpiyn;
	S[3] += dxpitn*vx - dxpixn + dypitn*vy - dypiyn + dnpitn*vn - dnpinn;
#else
	S[0] += dxpitt * vx - dxpitx - vx * dxPi + dypitt * vy - dypity - vy * dyPi + dnpitt * vn - dnpitn - vn * dnPi;
	S[1] += dxpitx * vx - dxpixx + dypitx * vy - dypixy + dnpitx * vn - dnpixn - dxPi;
	S[2] += dxpity * vx - dxpixy + dypity * vy - dypiyy + dnpity * vn - dnpiyn - dyPi;
	S[3] += dxpitn * vx - dxpixn + dypitn * vy - dypiyn + dnpitn * vn - dnpinn - dnPi / powf(t, 2.0f);
#endif

	//=========================================================
	// \pi^{\mu\nu} source terms
	//=========================================================
#ifndef IDEAL
	PRECISION pimunuRHS[NUMBER_DISSIPATIVE_CURRENTS];
	setPimunuSourceTerms(pimunuRHS, t, e, p, ut, ux, uy, un, utp, uxp, uyp, unp, pitt, pitx, pity, pitn, pixx, pixy, pixn, piyy, piyn, pinn, Pi, dxut, dyut,
			dnut, dxux, dyux, dnux, dxuy, dyuy, dnuy, dxun, dyun, dnun, dkvk);
	for (unsigned int n = 0; n < NUMBER_DISSIPATIVE_CURRENTS; ++n)
		S[n + 4] = pimunuRHS[n];
#endif
}
/***************************************************************************************************************************************************/

__device__
void loadSourceTermsX(const PRECISION * const __restrict__ I,
PRECISION * const __restrict__ S, const FLUID_VELOCITY * const __restrict__ u, int s) {
	//=========================================================
	// spatial derivatives of the conserved variables \pi^{\mu\nu}
	//=========================================================
	PRECISION facX = 1 / d_dx / 2;
	int ptr = 20; // 5 * n (with n = 4 corresponding to pitt)
	PRECISION dxpitt = (*(I + ptr + 3) - *(I + ptr + 1)) * facX;
	ptr += 5;
	PRECISION dxpitx = (*(I + ptr + 3) - *(I + ptr + 1)) * facX;
	ptr += 5;
	PRECISION dxpity = (*(I + ptr + 3) - *(I + ptr + 1)) * facX;
	ptr += 5;
	PRECISION dxpitn = (*(I + ptr + 3) - *(I + ptr + 1)) * facX;
	ptr += 5;
	PRECISION dxpixx = (*(I + ptr + 3) - *(I + ptr + 1)) * facX;
	ptr += 5;
	PRECISION dxpixy = (*(I + ptr + 3) - *(I + ptr + 1)) * facX;
	ptr += 5;
	PRECISION dxpixn = (*(I + ptr + 3) - *(I + ptr + 1)) * facX;
	ptr += 20;

	PRECISION ut = u->ut[s];
	PRECISION ux = u->ux[s];

	//=========================================================
	// set dx terms in the source terms
	//=========================================================
	PRECISION vx = ux / ut;
#ifndef PI
	S[0] = dxpitt*vx - dxpitx;
	S[1] = dxpitx*vx - dxpixx;
#else
	PRECISION dxPi = (*(I + ptr + 3) - *(I + ptr + 1)) * facX;
	S[0] = dxpitt * vx - dxpitx - vx * dxPi;
	S[1] = dxpitx * vx - dxpixx - dxPi;
#endif
	S[2] = dxpity * vx - dxpixy;
	S[3] = dxpitn * vx - dxpixn;
}

__device__
void loadSourceTermsY(const PRECISION * const __restrict__ J,
PRECISION * const __restrict__ S, const FLUID_VELOCITY * const __restrict__ u, int s) {
	//=========================================================
	// spatial derivatives of the conserved variables \pi^{\mu\nu}
	//=========================================================
	PRECISION facY = 1 / d_dy / 2;
	int ptr = 20; // 5 * n (with n = 4 corresponding to pitt)
	PRECISION dypitt = (*(J + ptr + 3) - *(J + ptr + 1)) * facY;
	ptr += 5;
	PRECISION dypitx = (*(J + ptr + 3) - *(J + ptr + 1)) * facY;
	ptr += 5;
	PRECISION dypity = (*(J + ptr + 3) - *(J + ptr + 1)) * facY;
	ptr += 5;
	PRECISION dypitn = (*(J + ptr + 3) - *(J + ptr + 1)) * facY;
	ptr += 10;
	PRECISION dypixy = (*(J + ptr + 3) - *(J + ptr + 1)) * facY;
	ptr += 10;
	PRECISION dypiyy = (*(J + ptr + 3) - *(J + ptr + 1)) * facY;
	ptr += 5;
	PRECISION dypiyn = (*(J + ptr + 3) - *(J + ptr + 1)) * facY;
	ptr += 10;

	PRECISION ut = u->ut[s];
	PRECISION uy = u->uy[s];

	//=========================================================
	// set dy terms in the source terms
	//=========================================================
	PRECISION vy = uy / ut;
#ifndef PI
	S[0] = dypitt*vy - dypity;
	S[2] = dypity*vy - dypiyy;
#else
	PRECISION dyPi = (*(J + ptr + 3) - *(J + ptr + 1)) * facY;
	S[0] = dypitt * vy - dypity - vy * dyPi;
	S[2] = dypity * vy - dypiyy - dyPi;
#endif
	S[1] = dypitx * vy - dypixy;
	S[3] = dypitn * vy - dypiyn;
}

__device__
void loadSourceTermsZ(const PRECISION * const __restrict__ K,
PRECISION * const __restrict__ S, const FLUID_VELOCITY * const __restrict__ u, int s, PRECISION t) {
	//=========================================================
	// spatial derivatives of the conserved variables \pi^{\mu\nu}
	//=========================================================
	PRECISION facZ = 1 / d_dz / 2;
	int ptr = 20; // 5 * n (with n = 4 corresponding to pitt)
	PRECISION dnpitt = (*(K + ptr + 3) - *(K + ptr + 1)) * facZ;
	ptr += 5;
	PRECISION dnpitx = (*(K + ptr + 3) - *(K + ptr + 1)) * facZ;
	ptr += 5;
	PRECISION dnpity = (*(K + ptr + 3) - *(K + ptr + 1)) * facZ;
	ptr += 5;
	PRECISION dnpitn = (*(K + ptr + 3) - *(K + ptr + 1)) * facZ;
	ptr += 15;
	PRECISION dnpixn = (*(K + ptr + 3) - *(K + ptr + 1)) * facZ;
	ptr += 10;
	PRECISION dnpiyn = (*(K + ptr + 3) - *(K + ptr + 1)) * facZ;
	ptr += 5;
	PRECISION dnpinn = (*(K + ptr + 3) - *(K + ptr + 1)) * facZ;
	ptr += 5;

	PRECISION ut = u->ut[s];
	PRECISION un = u->un[s];

	//=========================================================
	// set dn terms in the source terms
	//=========================================================
	PRECISION vn = un / ut;
#ifndef PI
	S[0] = dnpitt*vn - dnpitn;
	S[3] = dnpitn*vn - dnpinn;
#else
	PRECISION dnPi = (*(K + ptr + 3) - *(K + ptr + 1)) * facZ;
	S[0] = dnpitt * vn - dnpitn - vn * dnPi;
	S[3] = dnpitn * vn - dnpinn - dnPi / powf(t, 2);
#endif
	S[1] = dnpitx * vn - dnpixn;
	S[2] = dnpity * vn - dnpiyn;
}

__device__
void loadSourceTerms2(const PRECISION * const __restrict__ Q,
PRECISION * const __restrict__ S, const FLUID_VELOCITY * const __restrict__ u,
PRECISION utp, PRECISION uxp, PRECISION uyp, PRECISION unp,
PRECISION t, PRECISION e, const PRECISION * const __restrict__ pvec, int s) {
	//=========================================================
	// conserved variables
	//=========================================================
	PRECISION ttt = Q[0];
	PRECISION ttx = Q[1];
	PRECISION tty = Q[2];
	PRECISION ttn = Q[3];
#ifdef PIMUNU
	PRECISION pitt = Q[4];
	PRECISION pitx = Q[5];
	PRECISION pity = Q[6];
	PRECISION pitn = Q[7];
	PRECISION pixx = Q[8];
	PRECISION pixy = Q[9];
	PRECISION pixn = Q[10];
	PRECISION piyy = Q[11];
	PRECISION piyn = Q[12];
	PRECISION pinn = Q[13];
#else
	PRECISION pitt = 0;
	PRECISION pitx = 0;
	PRECISION pity = 0;
	PRECISION pitn = 0;
	PRECISION pixx = 0;
	PRECISION pixy = 0;
	PRECISION pixn = 0;
	PRECISION piyy = 0;
	PRECISION piyn = 0;
	PRECISION pinn = 0;
#endif
#ifdef PI
	PRECISION Pi = Q[14];
#else
	PRECISION Pi = 0;
#endif

	//=========================================================
	// primary variables
	//=========================================================
	PRECISION *utvec = u->ut;
	PRECISION *uxvec = u->ux;
	PRECISION *uyvec = u->uy;
	PRECISION *unvec = u->un;

	PRECISION p =  pvec[s];
	PRECISION ut = utvec[s];
	PRECISION ux = uxvec[s];
	PRECISION uy = uyvec[s];
	PRECISION un = unvec[s];

	int stride = d_ncx * d_ncy;

	//=========================================================
	// spatial derivatives of primary variables
	//=========================================================
	PRECISION facX = 1 / d_dx / 2;
	PRECISION facY = 1 / d_dy / 2;
	PRECISION facZ = 1 / d_dz / 2;

	#ifndef USE_APPROX_DERIVATIVE
	//use central finite difference

	// dx of u^{\mu} components
	PRECISION dxut = (*(utvec + s + 1) - *(utvec + s - 1)) * facX;
	PRECISION dxux = (*(uxvec + s + 1) - *(uxvec + s - 1)) * facX;
	PRECISION dxuy = (*(uyvec + s + 1) - *(uyvec + s - 1)) * facX;
	PRECISION dxun = (*(unvec + s + 1) - *(unvec + s - 1)) * facX;

	// dy of u^{\mu} components
	PRECISION dyut = (*(utvec + s + d_ncx) - *(utvec + s - d_ncx)) * facY;
	PRECISION dyux = (*(uxvec + s + d_ncx) - *(uxvec + s - d_ncx)) * facY;
	PRECISION dyuy = (*(uyvec + s + d_ncx) - *(uyvec + s - d_ncx)) * facY;
	PRECISION dyun = (*(unvec + s + d_ncx) - *(unvec + s - d_ncx)) * facY;

	// dn of u^{\mu} components
	PRECISION dnut = (*(utvec + s + stride) - *(utvec + s - stride)) * facZ;
	PRECISION dnux = (*(uxvec + s + stride) - *(uxvec + s - stride)) * facZ;
	PRECISION dnuy = (*(uyvec + s + stride) - *(uyvec + s - stride)) * facZ;
	PRECISION dnun = (*(unvec + s + stride) - *(unvec + s - stride)) * facZ;

	// pressure
	PRECISION dxp = (*(pvec + s + 1) - *(pvec + s - 1)) * facX;
	PRECISION dyp = (*(pvec + s + d_ncx) - *(pvec + s - d_ncx)) * facY;
	PRECISION dnp = (*(pvec + s + stride) - *(pvec + s - stride)) * facZ;

	#else
	//use minmod flux limiter to dampen oscillations in u and p

	PRECISION p_right =  pvec[s+1];
	PRECISION ut_right = utvec[s+1];
	PRECISION ux_right = uxvec[s+1];
	PRECISION uy_right = uyvec[s+1];
	PRECISION un_right = unvec[s+1];

	PRECISION p_left =  pvec[s-1];
	PRECISION ut_left = utvec[s-1];
	PRECISION ux_left = uxvec[s-1];
	PRECISION uy_left = uyvec[s-1];
	PRECISION un_left = unvec[s-1];

	PRECISION p_top =  pvec[s+d_ncx];
	PRECISION ut_top = utvec[s+d_ncx];
	PRECISION ux_top = uxvec[s+d_ncx];
	PRECISION uy_top = uyvec[s+d_ncx];
	PRECISION un_top = unvec[s+d_ncx];

	PRECISION p_bottom =  pvec[s-d_ncx];
	PRECISION ut_bottom = utvec[s-d_ncx];
	PRECISION ux_bottom = uxvec[s-d_ncx];
	PRECISION uy_bottom = uyvec[s-d_ncx];
	PRECISION un_bottom = unvec[s-d_ncx];

	PRECISION p_forward =  pvec[s+stride];
	PRECISION ut_forward = utvec[s+stride];
	PRECISION ux_forward = uxvec[s+stride];
	PRECISION uy_forward = uyvec[s+stride];
	PRECISION un_forward = unvec[s+stride];

	PRECISION p_backward =  pvec[s-stride];
	PRECISION ut_backward = utvec[s-stride];
	PRECISION ux_backward = uxvec[s-stride];
	PRECISION uy_backward = uyvec[s-stride];
	PRECISION un_backward = unvec[s-stride];

	// dx of u^{\mu} components
	PRECISION dxut = approximateDerivative(ut_left, ut, ut_right) / d_dx;
	PRECISION dxux = approximateDerivative(ux_left, ux, ux_right) / d_dx;
	PRECISION dxuy = approximateDerivative(uy_left, uy, uy_right) / d_dx;
	PRECISION dxun = approximateDerivative(un_left, un, un_right) / d_dx;

	// dy of u^{\mu} components
	PRECISION dyut = approximateDerivative(ut_bottom, ut, ut_top) / d_dy;
	PRECISION dyux = approximateDerivative(ux_bottom, ux, ux_top) / d_dy;
	PRECISION dyuy = approximateDerivative(uy_bottom, uy, uy_top) / d_dy;
	PRECISION dyun = approximateDerivative(un_bottom, un, un_top) / d_dy;

	//dn of u^{\mu} components
	PRECISION dnut = approximateDerivative(ut_backward, ut, ut_forward) / d_dz;
	PRECISION dnux = approximateDerivative(ux_backward, ux, ux_forward) / d_dz;
	PRECISION dnuy = approximateDerivative(uy_backward, uy, uy_forward) / d_dz;
	PRECISION dnun = approximateDerivative(un_backward, un, un_forward) / d_dz;

	// pressure
	PRECISION dxp = approximateDerivative(p_left, p, p_right) / d_dx;
	PRECISION dyp = approximateDerivative(p_bottom, p, p_top) / d_dy;
	PRECISION dnp = approximateDerivative(p_backward, p, p_forward) / d_dz;
	#endif


	//=========================================================
	// T^{\mu\nu} source terms
	//=========================================================
	PRECISION tnn = Tnn(e, p + Pi, un, pinn, t);
	PRECISION vx = ux / ut;
	PRECISION vy = uy / ut;
	PRECISION vn = un / ut;
	PRECISION dxvx = (dxux - vx * dxut) / ut;
	PRECISION dyvy = (dyuy - vy * dyut) / ut;
	PRECISION dnvn = (dnun - vn * dnut) / ut;
	PRECISION dkvk = dxvx + dyvy + dnvn;
	S[0] = -(ttt / t + t * tnn) + dkvk * (pitt - p - Pi) - vx * dxp - vy * dyp - vn * dnp;
	S[1] = -ttx / t - dxp + dkvk * pitx;
	S[2] = -tty / t - dyp + dkvk * pity;
	S[3] = -3 * ttn / t - dnp / powf(t, 2) + dkvk * pitn;
#ifdef USE_CARTESIAN_COORDINATES
	S[0] = dkvk*(pitt-p-Pi) - vx*dxp - vy*dyp - vn*dnp;
	S[1] = -dxp + dkvk*pitx;
	S[2] = -dyp + dkvk*pity;
	S[3] = -dnp + dkvk*pitn;
#endif

	//=========================================================
	// \pi^{\mu\nu} source terms
	//=========================================================
#ifndef IDEAL
	PRECISION pimunuRHS[NUMBER_DISSIPATIVE_CURRENTS];
	setPimunuSourceTerms(pimunuRHS, t, e, p, ut, ux, uy, un, utp, uxp, uyp, unp, pitt, pitx, pity, pitn, pixx, pixy, pixn, piyy, piyn, pinn, Pi, dxut, dyut,
			dnut, dxux, dyux, dnux, dxuy, dyuy, dnuy, dxun, dyun, dnun, dkvk);
	for (unsigned int n = 0; n < NUMBER_DISSIPATIVE_CURRENTS; ++n)
		S[n + 4] = pimunuRHS[n];
#endif
}
