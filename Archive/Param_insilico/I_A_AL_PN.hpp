//CREATED BY SHREYA FROM LN to PN -- changed parameters

#ifndef INCLUDED_I_A_ANTENNAL_LOBE_PN_HH_HPP
#define INCLUDED_I_A_ANTENNAL_LOBE_PN_HH_HPP

#include "insilico/core/engine.hpp"

namespace insilico {

class I_A {
 public:
  static void current(state_type &variables, state_type &dxdt, const double t, int index) {
    double ga = 10, ea = -95.0, Cels = 22;

    int v_index = engine::neuron_index(index, "v");
    int ma_index = engine::neuron_index(index, "ma");
    int ha_index = engine::neuron_index(index, "ha");

    double v = variables[v_index];
    double ma = variables[ma_index];
    double ha = variables[ha_index];

    double Tad = pow(3, ((Cels - 23.5)/10));

    double ma_inf = (1/(1+exp(-(v+60.0)/8.5)));
    double ha_inf = (1/(1+exp((v+78)/6)));
    double tau_ma = 1/(exp((v+35.82)/19.69) + exp(-(v+79.69)/12.7) + 0.37) / Tad;
    double tau_ha = 1/(exp((v+46.05)/5) + exp(-(v+238.4)/37.45)) / Tad;

    if (v >= -63.0) {
	tau_ha = 19 / Tad;
    } 

    // ODE set
    dxdt[ma_index]= (-1/tau_ma) * (ma - ma_inf);
    dxdt[ha_index]= (-1/tau_ha) * (ha - ha_inf);
   
    engine::neuron_value(index, "I_A", (ga * pow(ma,4) * ha * (v - ea)));

    } // function current
}; // class I_A

} // insilico

#endif


