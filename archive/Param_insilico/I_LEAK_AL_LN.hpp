#ifndef INCLUDED_I_LEAK_AL_LN_HPP
#define INCLUDED_I_LEAK_AL_LN_HPP

#include "insilico/core/engine.hpp"

namespace insilico {

class I_LEAK_LN_ {
 public:
  static void current(state_type &variables, state_type &dxdt, const double t, int index) {
    double gl = 0.15, el = -50;

    //Variables
    int v_index = engine::neuron_index(index, "v");
    double v = variables[v_index];

    // Current
    engine::neuron_value(index, "I_LEAK_LN", (gl * (v - el)));

  } // function current
}; // class I_LEAK_LN_

} // insilico

#endif
