"""
Geometric utilities for modeling energy deposition in microdosimetry.

This module provides helper functions to support radial dose integration
and geometric modeling of ion tracks intersecting sensitive volumes.

All methods assume cylindrical symmetry and operate in micrometer units.
"""

import numpy as np
from typing import Optional


DEFAULT_BASE_POINTS = 150


class GeometryTools:
    """
    Collection of geometric helper methods for particle interaction modeling.

    Includes utilities to compute sampling point densities, logarithmic radii distributions,
    and intersection areas between circular regions.
    """

    @staticmethod
    def calculate_squared_radius_ratio(
        numerator_radius: float,
        denominator_radius: float
    ) -> float:
        """
        Calculate the squared ratio between two strictly positive radii.

        This helper centralizes geometric radius scaling used across pyMKM,
        including nucleus-to-domain scaling in the MKM saturation parameter
        and domain-to-nucleus scaling in the MCF-MKM formulation.

        :param numerator_radius: Radius used in the numerator.
        :type numerator_radius: float
        :param denominator_radius: Radius used in the denominator.
        :type denominator_radius: float

        :returns: Squared radius ratio ``(numerator_radius / denominator_radius)^2``.
        :rtype: float

        :raises ValueError: If either radius is not strictly positive.
        """
        if numerator_radius <= 0 or denominator_radius <= 0:
            raise ValueError("Both radii must be positive.")

        return (numerator_radius / denominator_radius) ** 2

    @staticmethod
    def determine_sampling_points(energy: float, radius_max: float, base_points: int = DEFAULT_BASE_POINTS) -> int:
        """
        Determine the number of sampling points based on energy and target radius.
    
        The returned value scales with both energy and radius_max using a heuristic multiplier.
    
        :param energy: The particle's kinetic energy in MeV/u.
        :type energy: float
        :param radius_max: The maximum radius of interest in micrometers.
        :type radius_max: float
        :param base_points: Baseline number of sampling points (default is 150).
        :type base_points: int
    
        :returns: Total number of sampling points to use.
        :rtype: int
        """
        N0 = base_points

        if energy <= 10 or radius_max <= 0.05:
            multiplier = 20
        elif energy <= 20 or radius_max <= 0.1:
            multiplier = 15
        elif energy <= 40 or radius_max <= 0.5:
            multiplier = 12
        elif energy <= 60 or radius_max <= 1:
            multiplier = 8
        elif energy <= 80 or radius_max <= 2:
            multiplier = 6
        elif energy <= 100 or radius_max <= 5:
            multiplier = 5
        elif energy <= 150 or radius_max <= 10:
            multiplier = 4
        elif energy <= 200 or radius_max <= 20:
            multiplier = 3
        else:
            multiplier = 2

        return N0 * multiplier

    @staticmethod
    def generate_default_radii(energy: float, radius_max: float, radius_min: Optional[float] = 1e-3, base_points: int = DEFAULT_BASE_POINTS) -> np.ndarray:
        """
        Generate a logarithmically spaced array of radii for dose integration.
    
        The number of points is adjusted dynamically based on energy and radius_max.
        Requires both energy and radius_max to be specified.
    
        :param energy: The particle's kinetic energy in MeV/u.
        :type energy: float
        :param radius_max: The maximum radius in micrometers.
        :type radius_max: float
        :param radius_min: The minimum radius in micrometers (default is 1e-3 µm).
        :type radius_min: Optional[float]
        :param base_points: Baseline number of sampling points (default is 150).
        :type base_points: int
    
        :returns: Radii sampled on a log scale between radius_min and radius_max.
        :rtype: np.ndarray
    
        :raises ValueError: If `energy` or `radius_max` is not provided.
        """
        if energy is None or radius_max is None:
            raise ValueError("Both energy and maximum radius must be provided.")

        N = GeometryTools.determine_sampling_points(energy, radius_max, base_points)
        return np.logspace(np.log10(radius_min), np.ceil(np.log10(radius_max)), N)
    
    @staticmethod
    def calculate_intersection_area(r1: np.ndarray, r2: float, d: float) -> np.ndarray:
        """
        Calculate intersection areas between circles with variable radii and a fixed radius.
    
        For each radius in `r1`, the method computes the overlap area with a circle of radius `r2`,
        located at a center-to-center distance `d`.
    
        :param r1: Array of radii for the first set of circles.
        :type r1: np.ndarray
        :param r2: Radius of the second (fixed) circle.
        :type r2: float
        :param d: Distance between centers of the two circles.
        :type d: float
    
        :returns: Array of intersection areas for each radius in `r1`.
        :rtype: np.ndarray
        """
        r1 = np.asarray(r1).flatten()
        intersection_area = np.zeros_like(r1)

        non_overlap_mask = d >= (r1 + r2)
        intersection_area[non_overlap_mask] = 0.0

        contained_mask = d <= np.abs(r1 - r2)
        intersection_area[contained_mask] = np.pi * np.minimum(r1[contained_mask], r2)**2

        overlap_mask = ~non_overlap_mask & ~contained_mask
        if np.any(overlap_mask):
            r1_overlap = r1[overlap_mask]
            r1_sq = r1_overlap**2
            r2_sq = r2**2
            d_overlap = d

            arg1 = (d_overlap**2 + r1_sq - r2_sq) / (2 * d_overlap * r1_overlap)
            arg2 = (d_overlap**2 + r2_sq - r1_sq) / (2 * d_overlap * r2)
            arg1 = np.clip(arg1, -1.0, 1.0)
            arg2 = np.clip(arg2, -1.0, 1.0)

            part1 = r1_sq * np.arccos(arg1)
            part2 = r2_sq * np.arccos(arg2)
            part3 = 0.5 * np.sqrt(
                (-d_overlap + r1_overlap + r2) *
                (d_overlap + r1_overlap - r2) *
                (d_overlap - r1_overlap + r2) *
                (d_overlap + r1_overlap + r2)
            )

            intersection_area[overlap_mask] = part1 + part2 - part3

        return intersection_area
