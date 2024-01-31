"""
Simple geometry functions to calculate the equation of a line and the intersection point of two lines.
"""

import numpy as np

def line(point1: np.array, point2: np.array):
    """
    Get the equation of the line that goes through two points.
    
    Parameters:
    -----------
    :type point1: numpy.array
    :type point2: numpy.array
    
    Returns:
    --------
    :rtype: tuple
    
    """
    A = (point1[1] - point2[1])
    B = (point2[0] - point1[0])
    C = -(point1[0]*point2[1] - point2[0]*point1[1])
    return A, B, C


def intersection(line1: tuple, line2: tuple):
    """
    Get the intersection point of two lines.
    
    Parameters:
    -----------
    :type line1: tuple
    :type line2: tuple
    
    Returns:
    --------
    :rtype: numpy.array
    
    
    """

    A1, B1, C1 = line1
    A2, B2, C2 = line2

    D  = A1 * B2 - B1 * A2
    Dx = C1 * B2 - B1 * C2
    Dy = A1 * C2 - C1 * A2

    if D == 0:
        return None

    x = Dx / D
    y = Dy / D
    return np.array([x, y])