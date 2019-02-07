### IMPORTS ###

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import tensor_array_ops

import tensorflow as tf
import numpy as np

###########INTEGRATOR FRAMEWORK############

#1 Constraint Checks

def _check_input_types(t, y0): # Ensure input is Correct
    if not (y0.dtype.is_floating or y0.dtype.is_complex):
        raise TypeError('`y0` must have a floating point or complex floating point dtype')
    if not t.dtype.is_floating:
        raise TypeError('`t` must have a floating point dtype')
        
def _assert_increasing(t): # Check Time is Monotonous
    assert_increasing = control_flow_ops.Assert(math_ops.reduce_all(t[1:] > t[:-1]), ['`t` must be monotonic increasing'])
    return ops.control_dependencies([assert_increasing])

#2 Integrator Class

class _Integrator():
    
    def __init__(self,n_,F_b):
        self.n_ = n_
        self.F_b = F_b
        
    def integrate(self, evol_func, y0, time_grid): # iterator
        time_delta_grid = time_grid[1:] - time_grid[:-1]
        scan_func = self._make_scan_func(evol_func)
        y_grid = functional_ops.scan(scan_func, (time_grid[:-1], time_delta_grid),y0)
        return array_ops.concat([[y0], y_grid], axis=0)
    
    def _make_scan_func(self, evol_func): # stepper function
        
        def scan_func(y, t_dt): 
            
            n_ = self.n_
            F_b = self.F_b
            
            t,dt = t_dt
            
            if n_>0:
                
                
                dy = self._step_func(evol_func, t, dt, y)
                dy = math_ops.cast(dy, dtype=y.dtype)
                out = y + dy
                
                ## Operate on non-integral
                
                ft = y[-n_:]
                
                l = tf.zeros(tf.shape(ft),dtype=ft.dtype)
                l_ = t-ft
                
                z = tf.less(y[:n_],F_b)
                z_ = tf.greater_equal(out[:n_],F_b)
                
                df = tf.where(tf.logical_and(z,z_),l_,l)
                
                ft_ = ft+df
                
                return tf.concat([out[:-n_],ft_],0)

            else:
                dy = self._step_func(evol_func, t, dt, y)
                dy = math_ops.cast(dy, dtype=y.dtype)
                return y + dy
        
        return scan_func

    def _step_func(self, evol_func, t, dt, y):
        k1 = evol_func(y, t)
        half_step = t + dt / 2
        dt_cast = math_ops.cast(dt, y.dtype)

        k2 = evol_func(y + dt_cast * k1 / 2, half_step)
        k3 = evol_func(y + dt_cast * k2 / 2, half_step)
        k4 = evol_func(y + dt_cast * k3, t + dt)
        return math_ops.add_n([k1, 2 * k2, 2 * k3, k4]) * (dt_cast / 6)

#3 Integral Caller

def odeint_fixed(func, y0, t, n_, F_b):
    t = ops.convert_to_tensor(t, preferred_dtype=dtypes.float64, name='t')
    y0 = ops.convert_to_tensor(y0, name='y0')
    _check_input_types(t, y0)

    with _assert_increasing(t):
        return _Integrator(n_, F_b).integrate(func, y0, t)
