// CREATED BY ADITHYA RAJAGOPALAN ON 13.1.2016

#ifndef INCLUDED_I_CA_AL_LN_HPP
#define INCLUDED_I_CA_AL_LN_HPP

#include "insilico/core/engine.hpp"


namespace insilico {

class I_CA_LN_ {
 public:
  static void current(state_type &variables, state_type &dxdt, const double t, int index) {
    double gca = 3.0, eca = 140.0;
    
    // variables
    int v_index = engine::neuron_index(index, "v");
    int mca_index = engine::neuron_index(index, "mca");
    int hca_index = engine::neuron_index(index, "hca");

    double v = variables[v_index];
    double mca = variables[mca_index];
    double hca = variables[hca_index];

    double mca_inf = 1/(1+exp(-(v+20.0)/6.5));
    double tau_mca = 1.5;
    double hca_inf = 1/(1+exp((v+25.0)/12));
    double tau_hca = 0.3*exp((v-40.0)/13.0) + 0.002*exp((60.0-v)/29);

    // ODE set
    dxdt[mca_index]= (-1/tau_mca) * (mca - mca_inf);
    dxdt[hca_index]= (-1/tau_hca) * (hca - hca_inf);
    
    // Current
    engine::neuron_value(index, "I_CA", (gca * pow(mca,2) * hca * (v - eca)));

    } // function current
}; // class I_CA_LN_

} // insilico

#endif

