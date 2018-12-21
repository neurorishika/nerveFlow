#ifndef INCLUDED_I_KLEAK_ANTENNAL_LOBE_LN_HH_HPP
#define INCLUDED_I_KLEAK_ANTENNAL_LOBE_LN_HH_HPP

#include "insilico/core/engine.hpp"

namespace insilico {

class I_KLEAK_LN_ {
 public:
  static void current(state_type &variables, state_type &dxdt, const double t, int index) {
    double gkl = 0.02, ekl = -95;
    
    //Variables
    int v_index = engine::neuron_index(index, "v");
    double v = variables[v_index];

    // Current
    engine::neuron_value(index, "I_KLEAK_LN", (gkl * (v - ekl)));

  } // function current
}; // class I_KLEAK_LN_

} // insilico

#endif
