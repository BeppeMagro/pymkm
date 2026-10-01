"""
Specific-energy calculations used by MKM, SMK, and MCF-MKM.

This module defines :class:`SpecificEnergy`, which computes single-event
specific energy as a function of impact parameter and the dose-averaged
quantities required by the different MKM formulations. It also provides the
saturation corrections used by classical MKM and SMK.

The sensitive region is modeled as a water cylinder whose axis is parallel to
the ion trajectory. Radial dose profiles are integrated in impact-parameter
space. The same single-event machinery is reused by MCF-MKM for domain and,
when requested, explicitly integrated nucleus specific energies.

Example usage::

    from pymkm.physics import ParticleTrack, SpecificEnergy

    track = ParticleTrack(
        model_name="Kiefer-Chatterjee",
        energy=100.0,
        atomic_number=6,
        let=1000.0,
    )
    specific_energy = SpecificEnergy(track, region_radius=0.5)

    z1, b = specific_energy.single_event_specific_energy()
    z_bar = specific_energy.dose_averaged_specific_energy(z1, b)
"""

import time
import numpy as np
from typing import Optional, Tuple, Union
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pymkm.physics.particle_track import ParticleTrack
from pymkm.utils.geometry_tools import DEFAULT_BASE_POINTS, GeometryTools
from pymkm.utils.parallel import optimal_worker_count
from pymkm.utils.integration import integrate_1d

class SpecificEnergy:
    """
    Compute microdosimetric specific energy quantities from a single ion track.
    
    Supports the single-event and dose-averaged specific-energy calculations
    used by MKM, SMK, and MCF-MKM, together with the saturation quantities
    required by MKM/SMK. The sensitive region is modeled as a cylinder whose
    axis is parallel to the particle track.
    """

    def __init__(
        self, particle_track: ParticleTrack, region_radius: float
    ) -> None:
        """
        Initialize the specific energy calculator for a cylindrical sensitive region.
    
        :param particle_track: Radial dose profile of a single ion track, including penumbra radius.
        :type particle_track: ParticleTrack
        :param region_radius: Radius of the sensitive region (e.g. cell nucleus or domain), in micrometers.
        :type region_radius: float
    
        :raises TypeError: If `particle_track` is not a ParticleTrack instance.
        :raises ValueError: If `region_radius` is not strictly positive.
        """
        if not isinstance(particle_track, ParticleTrack):
            raise TypeError("particle_track must be an instance of ParticleTrack.")
        if region_radius <= 0:
            raise ValueError("region_radius must be positive.")

        self.track = particle_track
        self.region_radius = float(region_radius)
        self.penumbra_radius = self.track.penumbra_radius

    def single_event_specific_energy(
        self,
        impact_parameters: Optional[np.ndarray] = None,
        base_points_b: Optional[int] = None,
        base_points_r: Optional[int] = None,
        parallel: bool = False,
        return_time: bool = False
    ) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, float]]:
        """
        Compute z₁(b): single-event specific energy as a function of impact parameter.
       
        :param impact_parameters: Optional array of impact parameters b (in μm). 
        :type impact_parameters: Optional[np.ndarray]
        :param base_points_b: Number of sampling points for the b grid. 
        :type base_points_b: Optional[int]
        :param base_points_r: Number of radial integration points per b.
        :type base_points_r: Optional[int]
        :param parallel: If True, evaluates all b values in parallel using threads.
        :type parallel: bool
        :param return_time: If True, returns the computation time in seconds.
        :type return_time: bool
    
        :return: 
            - If `return_time` is False: tuple (z_array, b_array)
            - If `return_time` is True: tuple (z_array, b_array, elapsed_time)
    
            where:
            - z_array: specific energy per event at each b [Gy]
            - b_array: impact parameter values [μm]
            - elapsed_time: wall-clock time in seconds [s]
        :rtype: Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, float]]
    
        :raises ValueError: If impact_parameters are invalid.
        """
        start_time = time.perf_counter()

        if impact_parameters is None or (
            isinstance(impact_parameters, np.ndarray) and impact_parameters.size == 0
        ):
            b_max = self.region_radius + self.penumbra_radius
            default_b_pts = DEFAULT_BASE_POINTS
            impact_parameters = GeometryTools.generate_default_radii(
                energy=self.track.energy,
                radius_max=b_max,
                base_points=base_points_b if base_points_b is not None else default_b_pts
            )
            
        b_array = np.sort(np.asarray(impact_parameters).flatten())
        if parallel:
            workers = optimal_worker_count(b_array)
            func = partial(self._compute_z_single_b, base_points_r=base_points_r)
            with ThreadPoolExecutor(max_workers=workers) as executor:
                z_vals = list(executor.map(func, b_array))
        else:
            z_vals = [self._compute_z_single_b(b, base_points_r) for b in b_array]

        z_array = np.array(z_vals)
        elapsed_time = time.perf_counter() - start_time if return_time else None
        return (z_array, b_array, elapsed_time) if return_time else (z_array, b_array)

    def _compute_z_single_b(
        self, b: float, base_points_r: Optional[int] = None
    ) -> float:
        """
        Compute z₁(b): specific energy deposited in a single event at impact parameter b.
       
        :param b: Impact parameter from ion path to region center [μm].
        :type b: float
        :param base_points_r: Number of radial integration points. If None, defaults are used.
        :type base_points_r: Optional[int]
    
        :return: Specific energy z₁(b) deposited in the region [Gy].
        :rtype: float
        """
        r_min = max(1e-6, b - self.region_radius)                               
        r_max = min(b + self.region_radius, self.penumbra_radius)        
        default_r_pts = DEFAULT_BASE_POINTS
        r_array = GeometryTools.generate_default_radii(
            energy=self.track.energy,
            radius_max=r_max,
            radius_min=r_min,
            base_points=base_points_r if base_points_r is not None else default_r_pts
        )
        dose_profile, _ = self.track.initial_local_dose(radius=r_array)       
        values = dose_profile.ravel()[:-1]       
        areas = GeometryTools.calculate_intersection_area(r_array, self.region_radius, b)
        diff_areas = np.diff(areas)
        z_b = np.sum(values * diff_areas)
        return z_b / (np.pi * self.region_radius**2)

    @staticmethod
    def compute_saturation_parameter(
        domain_radius: float,
        nucleus_radius: float,
        beta0: float
    ) -> float:
        """
        Compute the saturation parameter z₀ for MKM overkill correction.
            
        :param domain_radius: Radius of the sensitive domain (e.g. sub-nuclear volume), in μm.
        :type domain_radius: float
        :param nucleus_radius: Radius of the cell nucleus, in μm.
        :type nucleus_radius: float
        :param beta0: β₀ coefficient of the LQ model at low LET (in Gy⁻²).
        :type beta0: float
    
        :return: Saturation parameter z₀ in Gy.
        :rtype: float
    
        :raises ValueError: If any input is not strictly positive.
        """
        if beta0 <= 0:
            raise ValueError("beta0 must be positive.")

        radius_ratio_squared = GeometryTools.calculate_squared_radius_ratio(
            nucleus_radius,
            domain_radius
        )
        return radius_ratio_squared / np.sqrt(
            beta0 * (1 + radius_ratio_squared)
        )

    def saturation_corrected_single_event_specific_energy(
        self,
        z0: float,
        z_array: np.ndarray,
        model: str = "square_root"
    ) -> np.ndarray:
        """
        Compute z′₁(b): saturation-corrected specific energy from z₁(b) using z₀.
    
        Applies overkill correction to single-event specific energy using either:
          - Square-root model: z′₁(b) = z₀ · sqrt(1 - exp(-[z₁(b)/z₀]²))
          - Quadratic model:   z₁_sat(b) = [z′₁(b)]² / z₁(b)
    
        :param z0: Saturation parameter z₀ (Gy).
        :type z0: float
        :param z_array: Array of uncorrected z₁(b) values (Gy).
        :type z_array: np.ndarray
        :param model: Saturation correction model to apply: 'square_root' or 'quadratic'.
        :type model: str
    
        :return: Array of corrected specific energy values: z′₁(b) or z₁_sat(b) (Gy).
        :rtype: np.ndarray
    
        :raises ValueError: If the model is not supported.
        """
        z_prime = z0 * np.sqrt(1 - np.exp(-(z_array ** 2) / (z0 ** 2)))
        if model == "square_root":
            return z_prime
        elif model == "quadratic":
            eps = 1e-12
            z_sat = np.zeros_like(z_array)
            mask = z_array > eps
            z_sat[mask] = (z_prime[mask] ** 2) / z_array[mask]
            return z_sat
        else:
            raise ValueError(f"Unsupported model '{model}'.")

    def dose_averaged_specific_energy(
        self,
        z_array: np.ndarray,
        b_array: np.ndarray,
        z_corrected: Optional[np.ndarray] = None,
        model: str = "square_root",
        integration_method: str = "trapz"
    ) -> float:
        """
        Compute dose-averaged specific energy: z̄ or z̄′.

        If no saturation correction is applied, computes:

            z̄ = [ ∫ z₁(b)² * b db ] / [ ∫ z₁(b) * b db ]

        If corrected values z′₁(b) are provided:
          - Square-root model: uses [z′₁(b)]² in the numerator
          - Quadratic model: uses z₁(b) or z₁_sat(b) in the numerator

        :param z_array: Uncorrected single-event specific energy values z₁(b) [Gy].
        :type z_array: np.ndarray
        :param b_array: Impact parameter values [μm], must be sorted.
        :type b_array: np.ndarray
        :param z_corrected: Optional corrected values (e.g. z′₁(b) or z₁_sat(b)).
        :type z_corrected: Optional[np.ndarray]
        :param model: Saturation model used if z_corrected is given ('square_root' or 'quadratic').
        :type model: str
        :param integration_method: Integration rule to use: 'trapz', 'simps', or 'quad'.
        :type integration_method: str

        :return: Dose-averaged specific energy [Gy].
        :rtype: float

        :raises ValueError: If model or integration method is invalid.
        """
    
        if z_corrected is None:
            numerator = integrate_1d(
                z_array ** 2 * b_array,
                b_array,
                method=integration_method
            )
        else:
            if model == "square_root":
                numerator = integrate_1d(
                    z_corrected ** 2 * b_array,
                    b_array,
                    method=integration_method
                )
            elif model == "quadratic":
                numerator = integrate_1d(
                    z_array * z_corrected * b_array,
                    b_array,
                    method=integration_method
                )
            else:
                raise ValueError("Model must be 'square_root' or 'quadratic' when providing z_corrected.")
    
        denom = integrate_1d(
            z_array * b_array,
            b_array,
            method=integration_method
        )
        return 0.0 if denom == 0 else numerator / denom