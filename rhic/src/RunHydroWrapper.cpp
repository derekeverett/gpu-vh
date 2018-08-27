//this is an example of how the HYDRO class can be instantiated in a larger program
//by default this file is compiled, which can then be run in the same way that the standalone executable
//given by Run.cpp was run...

//by default the Makefile produces the executable cpu-vh

#include "HydroWrapper.h"

int main(int argc, char **argv)
{
  //Declare an instance of HYDRO class
  HYDRO vh;

  //pass the initial energy density vector
  //vh.initialize_ed_from_vector(init_e);

  //run the hydro until freezeout
  vh.run_hydro(argc, argv);

  //save the fo surface info to a struct which can be passed to CF/Sampler module
  /*
  typedef struct
  {
     double tau, x, y, eta; //contravariant spacetime position
     double dat, dax, day, dan; //COVARIANT surface normal vector
     double ut, ux, uy, un; //contravariant flow velocity
     double E, T, P; //energy density, Temperature and Isotropic Thermal Pressure
     double pitt, pitx, pity, pitn, pixx, pixy, pixn, piyy, piyn, pinn; //contravariant components of shear stress tensor, or pi_perp^(\mu\nu) in case of VAH
     double bulkPi; //bulk pressure, or residual bulk pressure in case of VAH
     double muB, muS; //baryon chemical potential, strangeness chem. pot.
     double nB, Vt, Vx, Vy, Vn; //baryon number density, contravariant baryon diffusion current, or in case of VAH transverse baryon diffusion vector

     //quantities exclusive to VAH
     double PL; //longitudinal pressure
     double PT; //transverse pressure
     double Wt, Wx, Wy, Wn; //contraviariant longitudinal momentum diffusion current W^\mu = W_perpz^\mu
     double Lambda; // effective Temperature
     double aL, aT; // longitudinal and transverse momentum anisotropy parameters
     double upsilonB; //effective baryon chemical potential
     double nBL; //LRF longitudinal baryon diffusion

     double c0,c1,c2,c3,c4; // for vah every FO has different delta-f coefficients

  } FO_surf;

  FO_surf *surf;
  vh.save_fo_surf(surf);
  */
}
