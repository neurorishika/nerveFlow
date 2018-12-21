// CREATED BY SHREYA SOMETIME IN FEBRUARY OR MARCH 2018

#ifndef INCLUDED_I_K_ANTENNAL_LOBE_PN_HH_HPP
#define INCLUDED_I_K_ANTENNAL_LOBE_PN_HH_HPP

#include "insilico/core/engine.hpp"

namespace insilico {

class I_K {
 public:
  static void current(state_type &variables, state_type &dxdt, const double t, int index) {
    double gk = 10, ek = -95, vtr_k = -50, Cels = 22.0;

    int v_index = engine::neuron_index(index, "v");
    int n_index = engine::neuron_index(index, "n");

    double v = variables[v_index];
    double n = variables[n_index];

    double v2 = v - vtr_k;
    double phi = pow(3,((Cels-36)/10));

    double alpha3 = 0.02*(15 - v2)/(exp((15 - v2)/5) - 1);
    double beta3 = 0.5*exp((10 - v2)/40);

    double tau_n = (1/(alpha3 + beta3)) / phi;
    double n_inf = alpha3/(alpha3 + beta3);
    
    dxdt[n_index]= -(n - n_inf)/tau_n;

    engine::neuron_value(index, "I_K", (gk * pow(n,4) * (v - ek)));

    } // function current
}; // class I_K

} // insilico

#endif

// Equations for I_K from LN files by Adithya

    /*
    //Updating constants
    double vtr = -50.0;
    double v2 = v - vtr ; //This term scales the voltage value

    double alpha_n =  0.02*(15.0 - v2)/(exp((15.0 - v2)/5.0) - 1.0);//following the equations given in the html file
    double beta_n = 0.5*exp((10.0-v2) / 40.0);
    double phi = pow(3,((Cels-36)/10));

    // ODE set
    dxdt[n_index]=phi*(alpha_n*(1 - n)-beta_n * n);

    std::cout << "n : " << n << std::endl;
    std::cout << "I_K : " << (gk * pow(n,4) * (v2 - ek)) << std::endl;
    */
    
//Equations from the review -- 

    /*
    double alpha_n =  0.02*(15.0 - v2)/exp((10.0 - v2)/40);//following the equations given in the html file
    double beta_n = 0.5*exp((10.0 - v2)/40);
    double phi = pow(3,((22-Cels)/10));

    double n_inf = alpha_n/(phi*(alpha_n + beta_n));
    double tau_n = 1/(phi*(alpha_n + beta_n));

    // ODE set
    dxdt[n_index]= -(n - n_inf)/tau_n;

    // Current
    engine::neuron_value(index, "I_K_ANTENNAL_LOBE_PN_HH", (gk * pow(n,4) * (v - ek))); 
    */

//test equations

    /*
    double alpha_n = (0.1 - 0.01 * v) / (exp(1 - 0.1 * v) - 1.0);
    double beta_n  = 0.125 * exp(-v / 80.0);

    dxdt[n_index]=(alpha_n*(1 - n)-beta_n * n);
    */

// Equations from NcellNetwork2...

    /*
    
    G_K = 10;///////////////////////                                                                                                                                                                                                                                               
    G_Na = 100;/////////////////////                                                                                                                                                                                                                                               
    Vtr = -50;
    VtrK = -50;
    S1 = 0.32;
    S2 = 0.02;
    v2 = v - Vtr;
    v2K = v - VtrK;
    Phi = pow(3,((Cels-36)/10));
    Alpha1 = 0.32*(13 - v2)/(exp((13 - v2)/4) - 1);
    Beta1 = 0.28*(v2 - 40)/(exp((v2 - 40)/5) - 1);
    m0 = Alpha1/(Alpha1 + Beta1);

    Alpha2 = 0.128*exp((17 - v2)/18);
    Beta2 = 4/(exp((40 - v2)/5) + 1);
    h0 = Alpha2/(Alpha2 + Beta2);

    Alpha3 = 0.02*(15 - v2)/(exp((15 - v2)/5) - 1);
    Beta3 = 0.5*exp((10 - v2)/40);
    n0 = Alpha3/(Alpha3 + Beta3);     }
  void calc(double m, double h, double n, double &fm, double &fh, double &fn,
            double v, double x);
};

double INaK::E_K = -95, INaK::E_Na = 50, INaK::Cels = 22;

void INaK::calc(double m, double h, double n, double &fm, double &fh, double &fn,
                   double v, double x){
  v2 = v - Vtr;
  v2K = v - VtrK;
  iNa = G_Na*m*m*m*h*(v - E_Na);
  Alpha1 = S1*(13 - v2)/(exp((13 - v2)/4) - 1);
  Beta1 = 0.28*(v2 - 40)/(exp((v2 - 40)/5) - 1);
  tau_m = 1/(Alpha1 + Beta1) / Phi;
  m_inf = Alpha1/(Alpha1 + Beta1);

  Alpha2 = 0.128*exp((17 - v2)/18);
  Beta2 = 4/(exp((40 - v2)/5) + 1);
  tau_h = 1/(Alpha2 + Beta2) / Phi;
  h_inf = Alpha2/(Alpha2 + Beta2);

  fm = -(m - m_inf)/tau_m;
  fh = -(h - h_inf)/tau_h;

  iK = G_K* n*n*n*n*(v - E_K);
  Alpha3 = S2*(15 - v2K)/(exp((15 - v2K)/5) - 1);
  Beta3 = 0.5*exp((10 - v2K)/40);
  tau_n = 1/(Alpha3 + Beta3) / Phi;
  n_inf = Alpha3/(Alpha3 + Beta3);
   */
