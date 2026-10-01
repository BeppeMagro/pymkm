"""
Computation engine for MKTable.

This module defines the logic to compute microdosimetric quantities
for a set of ions based on MKM, SMK, or MCF-MKM using energy–LET tables.

It integrates track structure modeling and specific-energy calculation with
model-specific averaging, saturation correction, and optional oxygen-effect
scaling (OSMK 2023).
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Union
import time
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from dataclasses import asdict, replace

from pymkm.physics.particle_track import ParticleTrack
from pymkm.physics.specific_energy import SpecificEnergy
from pymkm.utils.parallel import optimal_worker_count
from pymkm.biology.oxygen_effect import (
    compute_osmk2023_radioresistance,
    compute_osmk2023_effective_parameters,
)
from pymkm.biology.mcf_model import (
    compute_scaled_nuclear_specific_energy,
    compute_mcf_correction_factor,
    compute_mcf_averaged_quantities,
)

from .core import MKTable

def _run_energy_let_task(func, args):
    """
    Internal wrapper for multiprocessing task execution.

    :param func: Callable function to execute.
    :type func: Callable
    :param args: Tuple of arguments for the function.
    :type args: tuple
    :return: Output of func(*args)
    """
    return func(*args)

def _build_worker_params(
    mktable: MKTable,
    integration_method: str = "trapz",
    domain_radius: Optional[float] = None,
    z0: Optional[float] = None
) -> dict:
    """
    Build the flattened parameter dictionary used by microdosimetric workers.

    :param mktable: Source MKTable instance.
    :type mktable: MKTable
    :param integration_method: Numerical integration method.
    :type integration_method: str
    :param domain_radius: Optional effective domain radius overriding the table value.
    :type domain_radius: float or None
    :param z0: Optional effective saturation parameter overriding the table value.
    :type z0: float or None

    :return: Flattened worker parameter dictionary.
    :rtype: dict
    """
    p = mktable.params
    return {
        "model_name": p.model_name,
        "core_radius_type": p.core_radius_type,
        "base_points_b": p.base_points_b,
        "base_points_r": p.base_points_r,
        "domain_radius": p.domain_radius if domain_radius is None else domain_radius,
        "nucleus_radius": p.nucleus_radius,
        "z0": p.z0 if z0 is None else z0,
        "alpha0": p.alpha0,
        "beta0": p.beta0,
        "use_stochastic_model": p.use_stochastic_model,
        "use_mcf_model": p.use_mcf_model,
        "mcf_nucleus_mode": p.mcf_nucleus_mode,
        "integration_method": integration_method,
    }


def _compute_for_energy_let_pair(
    params: dict,
    energy: float,
    let: float,
    atomic_number: int
) -> dict:
    """
    Compute specific energies for one (energy, LET) pair.

    :param params: Flattened MKTableParameters dictionary.
    :type params: dict
    :param energy: Kinetic energy per nucleon [MeV/u].
    :type energy: float
    :param let: Linear energy transfer [MeV/cm].
    :type let: float
    :param atomic_number: Atomic number of the ion.
    :type atomic_number: int

    :return: Dictionary of computed microdosimetric quantities.
    :rtype: dict
    """
    track = ParticleTrack(
        model_name=params["model_name"],
        core_radius_type=params["core_radius_type"],
        energy=energy,
        atomic_number=atomic_number,
        let=let,
        base_points=params["base_points_r"],
    )

    se_domain = SpecificEnergy(track, region_radius=params["domain_radius"])

    z_domain, b_domain = se_domain.single_event_specific_energy(
        base_points_b=params["base_points_b"],
        base_points_r=params["base_points_r"]
    )

    if params.get("use_mcf_model", False):
        if params.get("mcf_nucleus_mode", "scaled") == "scaled":
            z_nucleus = compute_scaled_nuclear_specific_energy(
                z_domain=z_domain,
                domain_radius=params["domain_radius"],
                nucleus_radius=params["nucleus_radius"]
            )
        elif params.get("mcf_nucleus_mode", "scaled") == "integrated":
            se_nucleus = SpecificEnergy(track, region_radius=params["nucleus_radius"])
            z_nucleus, _ = se_nucleus.single_event_specific_energy(
                impact_parameters=b_domain,
                base_points_r=params["base_points_r"]
            )
        else:
            raise ValueError(
                "mcf_nucleus_mode must be either 'scaled' or 'integrated'."
            )

        c = compute_mcf_correction_factor(
            z_domain=z_domain,
            z_nucleus=z_nucleus,
            alpha0=params["alpha0"],
            beta0=params["beta0"]
        )

        c_bar, z_bar_c = compute_mcf_averaged_quantities(
            z_domain=z_domain,
            b_array=b_domain,
            c=c,
            integration_method=params["integration_method"]
        )

        return {
            "c_bar": c_bar,
            "z_bar_c": z_bar_c
        }

    z_prime_domain = se_domain.saturation_corrected_single_event_specific_energy(
        z0=params["z0"], z_array=z_domain
    )

    z_bar_star_domain = se_domain.dose_averaged_specific_energy(
        z_array=z_domain,
        b_array=b_domain,
        z_corrected=z_prime_domain,
        integration_method=params["integration_method"]
    )

    result = {
        "z_bar_star_domain": z_bar_star_domain
    }

    if params["use_stochastic_model"]:
        z_bar_domain = se_domain.dose_averaged_specific_energy(
            z_array=z_domain,
            b_array=b_domain,
            integration_method=params["integration_method"]
        )

        se_nucleus = SpecificEnergy(track, region_radius=params["nucleus_radius"])

        z_nucleus, b_nucleus = se_nucleus.single_event_specific_energy(
            base_points_b=params["base_points_b"],
            base_points_r=params["base_points_r"]
        )

        z_bar_nucleus = se_nucleus.dose_averaged_specific_energy(
            z_array=z_nucleus,
            b_array=b_nucleus,
            integration_method=params["integration_method"]
        )

        result.update({
            "z_bar_domain": z_bar_domain,
            "z_bar_nucleus": z_bar_nucleus
        })

    return result

def _compute_for_ion(self: MKTable, ion: str, parallel: bool = True, number_of_workers: int = None, integration_method: str = "trapz"):
    """
    Compute all specific energies for a given ion in the table set.

    :param ion: Ion identifier (e.g., "C", "Carbon", 6).
    :type ion: str
    :param parallel: Whether to use parallel processing.
    :type parallel: bool
    :param number_of_workers: Optional user-defined number of workers.
    :type number_of_workers: int or None
    :param integration_method: Numerical integration method ('trapz', 'simps', or 'quad').
    :type integration_method: str

    :returns: Tuple of ion name and list of computed data entries.
    :rtype: tuple[str, list[dict]]
    """
    
    start_time = time.time()
    sp = self.sp_table_set.get(ion)
    energy_grid = sp.energy
    let_grid = sp.let

    job_list = [(E, L, sp.atomic_number) for E, L in zip(energy_grid, let_grid)]
    results = []
    
    # Prepare oxygen-corrected geometry if requested
    if self.params.apply_oxygen_effect:
        p = self.params
        _, f_rd, f_z0 = compute_osmk2023_radioresistance(
            K=p.K,
            pO2=p.pO2,
            Rmax=p.Rmax,
            f_rd_max=p.f_rd_max,
            f_z0_max=p.f_z0_max,
        )
        rd_eff, z0_eff = compute_osmk2023_effective_parameters(
            p.domain_radius, p.z0, f_rd, f_z0
        )
        
        print("✔ Using OSMK2023-corrected values:")
        print(f" - domain_radius: {self.params.domain_radius} → {rd_eff}")
        print(f" - z0: {self.params.z0} → {z0_eff}")
    else:
        rd_eff = self.params.domain_radius
        z0_eff = self.params.z0

    if parallel:
        worker_count = optimal_worker_count(job_list, user_requested=number_of_workers)
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            params_dict = _build_worker_params(
                self,
                integration_method=integration_method,
                domain_radius=rd_eff,
                z0=z0_eff,
            )
            func = partial(self._compute_for_energy_let_pair, params_dict)
            results = list(tqdm(
                executor.map(partial(_run_energy_let_task, func), job_list),
                total=len(job_list),
                desc=f"[{worker_count} workers] {sp.ion_name} ({sp.atomic_number},{sp.mass_number})",
                unit="energy"
            ))
    else:
        for args in tqdm(job_list, desc=f"{sp.ion_name} ({sp.atomic_number},{sp.mass_number})", unit="energy"):
            params_dict = _build_worker_params(
                self,
                integration_method=integration_method,
                domain_radius=rd_eff,
                z0=z0_eff,
            )
            results.append(self._compute_for_energy_let_pair(params_dict, *args))

    for job, result in zip(job_list, results):
        E, L, _ = job
        result.update({"energy": E, "let": L})

    elapsed = time.time() - start_time
    print(f"\n  ... completed in {elapsed:.2f} seconds.\n")
    return ion, results

def compute(
    self: MKTable,
    ions: Optional[List[Union[str, int]]] = None,
    energy: Optional[Union[float, List[float], np.ndarray]] = None,
    parallel: bool = True,
    number_of_workers: Optional[int] = None,
    integration_method: str = "trapz"
) -> None:
    """
    Compute per-ion microdosimetric tables using MKM, SMK, or MCF-MKM.

    For each ion:
      - Retrieves energy–LET grid
      - Computes specific energy and dose-averaged quantities
      - Applies model-specific averaging, saturation correction, and optional OSMK
      - Aggregates into a structured table

    :param self: MKTable instance.
    :type self: pymkm.mktable.core.MKTable
    :param ions: Ion identifiers to compute. If None, all available ions are used.
    :type ions: list[str or int], optional
    :param energy: Custom energy grid for resampling (if any).
    :type energy: float or list or np.ndarray, optional
    :param parallel: Whether to enable multiprocessing.
    :type parallel: bool
    :param number_of_workers: Optional user-defined number of workers.
    :type number_of_workers: int or None
    :param integration_method: Integration scheme ('trapz', 'simps', 'quad').
    :type integration_method: str

    :raises RuntimeError: If MKTable is not initialized.
    """
    
    if not self.sp_table_set or not self.params:
        raise RuntimeError("MKTable is not properly initialized.")

    ions = ions or self.sp_table_set.get_available_ions()
    ions = [self.sp_table_set._map_to_fullname(ion) for ion in ions]

    if energy is not None:
        custom_energy = np.atleast_1d(np.array(energy, dtype=float))
        self.sp_table_set.resample_all(custom_energy)

    original_params = replace(self.params)

    if not self.params.use_mcf_model:
        z0 = self.params.z0 or SpecificEnergy.compute_saturation_parameter(
            domain_radius=self.params.domain_radius,
            nucleus_radius=self.params.nucleus_radius,
            beta0=self.params.beta0
        )
        self.params.z0 = round(z0, 2)

    self._refresh_parameters(original_params)

    print("\nStarting table computation in {} mode ...".format("parallel" if parallel else "serial"))
    print(f"\nIons to be computed: {', '.join(ions)}\n")
    overall_start = time.time()
    results = {}
    for ion in ions:
        ion_key = self.sp_table_set._map_to_fullname(ion)
        ion, ion_results = self._compute_for_ion(ion, parallel=parallel, integration_method=integration_method, number_of_workers=number_of_workers)
        results[ion_key] = ion_results

    enriched_results = {}
    for ion_key, data in results.items():
        sp = self.sp_table_set.get(ion_key)
        rows = []
        for entry in data:
            if self.params.use_mcf_model:
                row = {
                    "energy": entry["energy"],
                    "let": entry["let"],
                    "c_bar": entry.get("c_bar"),
                    "z_bar_c": entry.get("z_bar_c")
                }
            else:
                row = {
                    "energy": entry["energy"],
                    "let": entry["let"],
                    "z_bar_star_domain": entry.get("z_bar_star_domain"),
                    "z_bar_domain": entry.get("z_bar_domain"),
                    "z_bar_nucleus": entry.get("z_bar_nucleus")
                }
            rows.append(row)

        df = pd.DataFrame(rows).sort_values("energy").reset_index(drop=True)

        sp_metadata = sp.to_dict()
        sp_metadata.pop("energy", None)
        sp_metadata.pop("let", None)

        enriched_results[ion_key] = {
            "params": asdict(self.params),
            "stopping_power_info": sp_metadata,
            "data": df
        }

    print("\n... finalizing results and updating ...")
    sorted_keys = sorted(
        enriched_results.keys(),
        key=lambda k: enriched_results[k]["stopping_power_info"].get("atomic_number")
        )
    self.table = {k: enriched_results[k] for k in sorted_keys}

    total_elapsed = time.time() - overall_start
    print(f"\n... done. Total elapsed time: {total_elapsed:.2f} seconds.")

MKTable.compute = compute
MKTable._compute_for_energy_let_pair = staticmethod(_compute_for_energy_let_pair)
MKTable._compute_for_ion = _compute_for_ion
