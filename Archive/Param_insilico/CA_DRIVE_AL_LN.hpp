// CREATED BY ADITHYA RAJAGOPALAN ON 14.1.2016 

#ifndef INCLUDED_CA_DRIVE_ANTENNAL_LOBE_LN_HH_MBMS_HPP
#define INCLUDED_CA_DRIVE_ANTENNAL_LOBE_LN_HH_MBMS_HPP

#include "insilico/core/engine.hpp"
#include "I_CA_AL_LN.hpp"


namespace insilico {

class CA_DRIVE_ {
 public:
  static void current(state_type &variables, state_type &dxdt, const double t, int index) {
    double Ca_inf = 2.4 * pow(10,-4);
    double A = 2 * pow(10,-4);
    double tau = 150;
    
    //Calling calcium concentration Initial value from n sets
    int cai_index = engine::neuron_index(index, "CA_DRIVE"); // cai is the calcium concentration
    double cai = variables[cai_index];
    
    //Calling Calcium current from CA Channel
    double I_Ca = engine::neuron_value(index, "I_CA"); 

    //ODE set
    dxdt[cai_index] = (-A * (I_Ca)) - ((cai - Ca_inf)/tau);

    engine::neuron_value(index, "CA_DRIVE", cai);

    } // function current
}; // class CA_DRIVE_ANTENNAL_LOBE_LN_HH_MBMS

} // insilico  

#endif
