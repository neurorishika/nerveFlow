### IMPORTS ###
import numpy as np

###########INTEGRATOR FRAMEWORK############


#1 Constraint Checks

def check_type(y,t): # Ensure Input is Correct
    return y.dtype == np.floating and t.dtype == np.floating

class _Integrator():

    def __init__(self,n_,F_b):
        self.n_ = n_
        self.F_b = F_b
    
    def integrate(self,func,y0,t):
        time_delta_grid = t[1:] - t[:-1]
        
        y = np.zeros((y0.shape[0],t.shape[0]))
        y[:,0] = y0

        for i in range(time_delta_grid.shape[0]):
            k1 = func(y[:,i], t[i])                               # RK4 Integration Steps
            half_step = t[i] + time_delta_grid[i] / 2
            k2 = func(y[:,i] + time_delta_grid[i] * k1 / 2, half_step)
            k3 = func(y[:,i] + time_delta_grid[i] * k2 / 2, half_step)
            k4 = func(y[:,i] + time_delta_grid[i] * k3, t + time_delta_grid[i])
            dy = (k1 + 2 * k2 + 2 * k3 + k4) * (time_delta_grid[i] / 6)
            out = dy + y[:,i]
            
            ft = y[-n_:,i]
            l = np.zeros(ft.shape)
            l_ = t[i]-ft
            z = y[:n_,i] < F_b
            z_ = out[:n_] > F_b
            df = np.where(np.logical_and(z,z_),[l_,l])
            ft_ = ft+df
            
            y[:,i+1] = np.concatenate([out[:-n_],ft_],0)
        return y


def odeint_fixed(func,y0,t,n_,F_b):
    y0 = np.array(y0)
    t = np.array(t)
    if check_type(y0,t):
        return _Integrator(n_, F_b).integrate(func,y0,t)
    else:
        print("error encountered")

