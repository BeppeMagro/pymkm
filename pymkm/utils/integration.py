"""
Shared one-dimensional numerical integration utilities.

This module centralizes the integration methods used by pyMKM when
computing impact-parameter averages. Supported methods intentionally
mirror the historical implementation in :mod:`pymkm.physics.specific_energy`.
"""

import numpy as np
from scipy.integrate import simpson, quad
from scipy.interpolate import interp1d


def integrate_1d(
    y_array: np.ndarray,
    x_array: np.ndarray,
    method: str = "trapz"
) -> float:
    """
    Integrate a one-dimensional sampled quantity.

    :param y_array: Sampled dependent-variable values.
    :type y_array: np.ndarray
    :param x_array: Sample positions associated with ``y_array``.
    :type x_array: np.ndarray
    :param method: Integration rule: ``"trapz"``, ``"simps"``, or ``"quad"``.
    :type method: str

    :return: Numerical integral over the sampled interval.
    :rtype: float

    :raises ValueError: If the requested integration method is unsupported.
    """
    if method == "trapz":
        # NumPy >= 2.0 exposes ``trapezoid``; older supported versions use
        # ``trapz``. Keep the public pyMKM method name unchanged while
        # avoiding the NumPy deprecation warning when possible.
        if hasattr(np, "trapezoid"):
            return np.trapezoid(y_array, x_array)
        return np.trapz(y_array, x_array)
    elif method == "simps":
        return simpson(y=y_array, x=x_array)
    elif method == "quad":
        f_interp = interp1d(x_array, y_array, kind="cubic", fill_value="extrapolate")
        result, _ = quad(f_interp, x_array[0], x_array[-1], limit=100)
        return result
    else:
        raise ValueError(f"Unsupported integration method: '{method}'")
