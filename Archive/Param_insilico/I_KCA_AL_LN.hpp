//CREATED BY ADITHYA RAJAGOPALAN ON 12.1.2016

#ifndef INCLUDED_I_KCA_AL_LN_HPP
#define INCLUDED_I_KCA_AL_LN_HPP

#include "insilico/core/engine.hpp"
#include "CA_DRIVE_AL_LN.hpp"

namespace insilico {

class I_KCA_LN_ {
 public:
  static void current(state_type &variables, state_type &dxdt, const double t, int index) {
    double gkca = 0.3, ekca = -90.0, Ra = 0.01, Rb= 0.02, Q = 2.3, Cels = 23.0;

    //variables
    int v_index = engine::neuron_index(index, "v");
    int mk_index = engine::neuron_index(index, "mk");
    
    double v = variables[v_index];
    double mk = variables[mk_index];

    // Calling the calcium current from the CA_DRIVE_ANTENAL_LOBE_LN_HH code
    double cai = engine::neuron_value(index, "CA_DRIVE"); 
    
    // Updating Constants
    double alpha = Ra * (cai);
    double beta = Rb;
    double Tad = pow(Q,((Cels-23)/10)); 

    double mk_inf = alpha/(alpha + beta);
    double tau_mk = 1/((alpha + beta) * Tad);

    // ODE set
    dxdt[mk_index]= (-1/tau_mk) * (mk - mk_inf); 

    // Current
    engine::neuron_value(index, "I_KCA", (gkca * Tad * mk * (v - ekca)));

    } // function current
}; // class I_KCA_LN_

} // insilico

#endif


    //double cai = engine::neuron_value(index, "CA_DRIVE_ANTENNAL_LOBE_LN_HH"); 
    // Calling the calcium current from the CA_DRIVE_ANTENAL_LOBE_LN_HH_MBMS code
