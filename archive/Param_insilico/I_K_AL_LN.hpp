//CREATED BY SHREYA ON 23 MARCH 2018

#ifndef INCLUDED_I_K_AL_LN_HPP
#define INCLUDED_I_K_AL_LN_HPP

#include "insilico/core/engine.hpp"

namespace insilico {

class I_K_LN_ {
 public:
  static void current(state_type &variables, state_type &dxdt, const double t, int index) {
    double gk = 10.0, ek = -95.0, vtr_k = -50, Cels = 22.0;

    //variables
    int v_index = engine::neuron_index(index, "v");
    int nk_index = engine::neuron_index(index, "nk");

    double v = variables[v_index];
    double nk = variables[nk_index];

    double v2 = v - vtr_k; //This term scales the voltage value

    double alpha_nk =  0.02*(15.0 - v2)/(exp((15.0 - v2)/5.0) - 1.0);
    double beta_nk = 0.5*exp((10.0-v2) / 40.0);
    double phi = pow(3,((Cels-36)/10));

    double tau_nk = 1/(alpha_nk + beta_nk) / phi;
    double nk_inf = alpha_nk/(alpha_nk+ beta_nk);

    // ODE set
    dxdt[nk_index] = -(nk - nk_inf)/tau_nk;

    // Current
    engine::neuron_value(index, "I_K_LN", (gk * pow(nk,4) * (v - ek)));

    } // function current
}; // class I_K_LN_

} // insilico

#endif

