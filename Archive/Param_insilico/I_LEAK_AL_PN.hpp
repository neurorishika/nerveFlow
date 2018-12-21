#ifndef INCLUDED_I_LEAK_ANTENNAL_LOBE_PN_HH_HPP
#define INCLUDED_I_LEAK_ANTENNAL_LOBE_PN_HH_HPP

#include "insilico/core/engine.hpp"

namespace insilico {

class I_LEAK {
 public:
  static void current(state_type &variables, state_type &dxdt, const double t, int index) {
    double gl = 0.15, el = -55;

    int v_index = engine::neuron_index(index, "v");
    double v = variables[v_index];

    engine::neuron_value(index, "I_LEAK", (gl * (v - el)));

  } // function current
}; // class I_LEAK_ANTENNAL_LOBE_LN_HH

} // insilico

#endif
