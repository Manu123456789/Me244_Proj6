"""
Surface functions for LIDAR simulation
Contains surface geometry definitions and gradient calculations

FILL IN THE BLANK SECTIONS MARKED WITH TODO
"""

import numpy as np

###################################################################################################
############################### Lidar Constants and Parameters ####################################
###################################################################################################

A_ = [0.04, 0.16, 0.64]                         # surface amplitudes

L1 = 1                                          # surface oscillation height
L2 = 1                                          # surface oscillation height
omega1 = 2                                      # surface frequency 1
omega2 = 1                                      # surface frequency 2


def surfG(x1, x2, A):
    """
    Surface height function - Equation 1
    
    Defines the height of the surface as a function of x and y positions.
    The surface is a sinusoidal oscillation in both x and y directions.
    
    Parameters
    ----------
    x1 : float or ndarray
        x position(s) of point(s)
    x2 : float or ndarray
        y position(s) of point(s)
    A : float
        surface amplitude parameter
        
    Returns
    -------
    float or ndarray
        Height of surface at given (x1, x2) position(s)
        
    FILL IN: Implement the surface height equation (eqn. 1)
    """
    # TODO: Uncomment and complete this line
    return 2 + A * np.sin(omega1 * x1 / L1) * np.sin(omega2 * x2 / L2)



def gradG(x1, x2, A):
    """
    Gradient of surface height function - Equation 1
    
    Computes the partial derivatives of the surface height with respect to x and y.
    Used for computing surface normal vectors.
    
    Parameters
    ----------
    x1 : float or ndarray
        x position(s) of point(s)
    x2 : float or ndarray
        y position(s) of point(s)
    A : float
        surface amplitude parameter
        
    Returns
    -------
    list
        [dG/dx1, dG/dx2] - partial derivatives in x and y directions
        
    FILL IN: Compute the partial derivatives
    Hint: Use chain rule and derivative of sin(u) = cos(u) * du/dx
    """
    # TODO: Compute gradGx = delG/delx_1
    gradGx = A * ((2 * omega1 * np.pi) / L1) * np.cos((2 * omega1 * np.pi * x1) / L1) * np.sin((2 * omega2 * np.pi * x2) / L2)

    # TODO: Compute gradGy = delG/delx_2
    gradGy = A * ((2 * omega2 * np.pi) / L2) * np.sin((2 * omega1 * np.pi * x1) / L1) * np.cos((2 * omega2 * np.pi * x2) / L2)

    return [gradGx, gradGy]
