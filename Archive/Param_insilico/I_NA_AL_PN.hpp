// CREATED BY SHREYA SOMETIME IN FEBRUARY OR MARCH 2018

#ifndef INCLUDED_I_NA_AL_PN_HPP
#define INCLUDED_I_NA_AL_PN_HPP

#include "insilico/core/engine.hpp"

namespace insilico {

class I_NA {
 public:
  static void current(state_type &variables, state_type &dxdt, const double t, int index) {
    double gna = 100, ena = 50, vtr_na = -50, Cels = 22;
    
    int v_index = engine::neuron_index(index, "v");
    int m_index = engine::neuron_index(index, "m");
    int h_index = engine::neuron_index(index, "h");

    double v = variables[v_index];
    double m = variables[m_index];
    double h = variables[h_index];

    double v2 = v - vtr_na;
    double phi = pow(3,((Cels-36)/10));
    
    double alpha1 = 0.32*(13 - v2)/(exp((13 - v2)/4) - 1);
    double beta1 = 0.28*(v2 - 40)/(exp((v2 - 40)/5) - 1);

    double alpha2 = 0.128*exp((17 - v2)/18);
    double beta2 = 4/(exp((40 - v2)/5) + 1);

    double tau_m = (1/(alpha1 + beta1)) / phi;
    double m_inf = alpha1/(alpha1 + beta1);

    double tau_h = (1/(alpha2 + beta2)) / phi;
    double h_inf = alpha2/(alpha2 + beta2);

    dxdt[m_index] = -(m - m_inf)/tau_m;
    dxdt[h_index] = -(h - h_inf)/tau_h;

    double ina = gna * pow(m,3) * h * (v - ena);

    engine::neuron_value(index, "I_NA", ina);

    } // function current
}; // class I_NA

} // insilico

#endif
