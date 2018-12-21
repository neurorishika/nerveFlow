#ifndef INCLUDED_I_KLEAK_ANTENNAL_LOBE_PN_HH_HPP
#define INCLUDED_I_KLEAK_ANTENNAL_LOBE_PN_HH_HPP

#include "insilico/core/engine.hpp"

namespace insilico {

class I_KLEAK {
 public:
  static void current(state_type &variables, state_type &dxdt, const double t, int index) {
   double gkl = 0.05, ekl = -95;
    
    int v_index = engine::neuron_index(index, "v");
    double v = variables[v_index];

    engine::neuron_value(index, "I_KLEAK", (gkl * (v - ekl)));

  } // function current
}; // class I_KLEAK

} // insilico

#endif
