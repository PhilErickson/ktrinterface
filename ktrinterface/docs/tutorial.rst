Tutorial
========

The main function provided by this module is :code:`ktrsolve`, a high level python interface to the KNITRO solver. While :code:`ktrsolve` allows for a high level of customization, it defaults based on the input provided by the programmer. In this spirit, it will take both constrained and unconstrained optimization problems, based on input.

For a basic example, consider minimizing the function :math:`f(x) = x^2` with initial guess :math:`x_0=2`. The solution can be obtained by calling `ktrsolve` as follows

.. code-block:: python

    from __future__ import print_function, division
    import ktrinterface as ki
    
    def fun(x):
        return x**2
    
    result = ki.ktrsolve(fun=fun, guess=2.0)
    print('Solution:', result['coef'])
    # Solution: [  9.16711151e-12]

While :code:`ktrsolve` requires a minimum of two arguments, one for the function definition and one for the initial guess, any options can be defined for the problem that are provided in the usual KNITRO callable library. These are given as a dictionary of :code:`options`. For example, to supress output from the optimization routine, the option :code:`outlev` can be set to :code:`none` by passing it through to :code:`options`

.. code-block:: python

    result = ki.ktrsolve(fun=fun, guess=2.0, options={'outlev': 'none'})

All possible problem options are given in the KNITRO User Manual under section 3.3.3 "KNITRO user options". Note that, while the classifier :code:`none` was given here for :code:`outlev`, the same option could also have been given the value of :code:`0` or, since all KNITRO-defined constants are made available through the module, :code:`ki.KTR_OUTLEV_NONE`.

To see a more complicated example, consider the first example problem in the KNITRO90 User Manual. 

.. math::
    
    & \underset{x}{\text{min   }}
    & & 1000 - x_1^2 - 2x_2^2 - x_3^2 - x_1x_2 - x_1x_3 \\
    & \text{s.t.   }
    & & 8x_1 + 14x_2 + 7x_3 - 56 = 0 \\
    &&& x_1^2 + x_2^2 + x_3^2 - 25 \geq 0

In this case, each constraint should be stored as an element of a NumPy array and bounds on the constraints are defined by KNITRO constants.

.. code-block:: python

    import numpy as np
    
    def fun(x):
        """ Define objective """
        return 1000 - x[0]**2 - 2*x[1]**2 - x[2]**2 - x[0]*x[1] - x[0]*x[2]
    
    def constr(x):
        """ Define constraints """
        return np.array([
            8*x[0] + 14*x[1] + 7*x[2] - 56,
            x[0]**2 + x[1]**2 + x[2]**2 -25
            ])
    
    # Define bounds for constraints and constraint types
    ktr_defs = {
        'bounds_low': np.array([0, 0]),
        'bounds_up': np.array([0, ki.KTR_INFBOUND]),
        'c_type': np.array([ki.KTR_CONTYPE_LINEAR, ki.KTR_CONTYPE_QUADRATIC])
        }
    
    # Initial guess
    x_init = np.array([2, 2, 2])
    
    # Call routine
    result = ki.ktrsolve(fun=fun, guess=x_init, constr=constr,
                         ktr_defs=ktr_defs)
    print("Result:" result['coef'])
    


