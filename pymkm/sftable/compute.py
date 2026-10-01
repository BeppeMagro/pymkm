"""
Computation of survival fraction (SF) curves from microdosimetric input.

This module defines the method :meth:`SFTable.compute`, which calculates SF
values over a dose grid based on results from an MKTable.

Supported models include:

- MKM (classic)
- SMK (stochastic)
- OSMK 2021/2023 (stochastic with oxygen correction)
- MCF-MKM

The method supports reuse of precomputed MKTable data, direct recomputation,
and model-specific oxygen corrections.
"""

from .core import SFTable
from typing import Union, Literal, Optional
import numpy as np
import pandas as pd
from pymkm.mktable.compute import _compute_for_energy_let_pair, _build_worker_params
from pymkm.physics.specific_energy import SpecificEnergy
from pymkm.biology.oxygen_effect import (
    compute_osmk2021_radioresistance,
    compute_osmk2023_radioresistance,
    compute_osmk_lq_coefficients,
    compute_osmk2023_effective_parameters,
)
from pymkm.biology.mkm_model import (
    compute_mkm_lq_coefficients,
    compute_smk_lq_coefficients,
    compute_smk_gamma,
)
from pymkm.biology.mcf_model import compute_mcf_lq_coefficients


_MICRODOSIMETRIC_COLUMNS = {
    "classic": ("z_bar_star_domain",),
    "stochastic": ("z_bar_star_domain", "z_bar_domain", "z_bar_nucleus"),
    "mcf": ("c_bar", "z_bar_c"),
}


def _find_precomputed_rows(
    ion_data: pd.DataFrame,
    *,
    energy: Optional[float] = None,
    let: Optional[float] = None,
) -> pd.DataFrame:
    """Return MKTable rows matching the requested energy and/or LET."""
    mask = np.ones(len(ion_data), dtype=bool)
    if energy is not None:
        mask &= np.isclose(
            ion_data["energy"].to_numpy(dtype=float),
            float(energy),
            rtol=1e-9,
            atol=1e-12,
        )
    if let is not None:
        mask &= np.isclose(
            ion_data["let"].to_numpy(dtype=float),
            float(let),
            rtol=1e-9,
            atol=1e-12,
        )
    return ion_data.loc[mask]


def _precomputed_result(row: pd.Series, model: str) -> Optional[dict]:
    """Extract model-specific microdosimetric quantities from one MKTable row."""
    required = _MICRODOSIMETRIC_COLUMNS[model]
    if any(column not in row.index or pd.isna(row[column]) for column in required):
        return None
    return {column: row[column] for column in required}


def compute(
    self: SFTable,
    *,
    ion: Union[str, int],
    energy: Optional[float] = None,
    let: Optional[float] = None,
    force_recompute: Optional[bool] = True,
    model: Optional[Literal["classic", "stochastic", "mcf"]] = None,
    apply_oxygen_effect: bool = False,
) -> None:
    """
    Compute survival fraction curve(s) based on microdosimetric inputs.

    If ``force_recompute`` is False, matching microdosimetric quantities already
    available in the associated MKTable are reused. Otherwise, or if no usable
    matching row exists, the quantities are computed directly.

    This function stores the results in ``self.table`` as a list of result dictionaries.

    :param ion: Ion name, symbol, or atomic number.
    :type ion: Union[str, int]
    :param energy: Kinetic energy per nucleon [MeV/u]. LET is inferred if omitted.
    :type energy: Optional[float]
    :param let: Linear energy transfer [MeV/cm]. Energy is inferred if omitted.
    :type let: Optional[float]
    :param force_recompute: If True, recompute even when matching MKTable data exist.
    :type force_recompute: Optional[bool]
    :param model: Microdosimetric model to use: "classic", "stochastic", or "mcf".
    :type model: Literal["classic", "stochastic", "mcf"], optional
    :param apply_oxygen_effect: Whether to apply OSMK. Valid only for stochastic mode.
    :type apply_oxygen_effect: bool

    :returns: None. Results are stored in ``self.table``.

    :raises ValueError: If inputs are inconsistent, or if required parameters are missing.
    """

    params = self.params
    model = model or params.mktable.model_version

    if model not in ("classic", "stochastic", "mcf"):
        raise ValueError("model must be one of: 'classic', 'stochastic', or 'mcf'.")

    if apply_oxygen_effect and model != "stochastic":
        raise ValueError("Oxygen effect (OSMK) can only be applied with model='stochastic'.")

    # Determine if OSMK effect is active.
    is_osmk = apply_oxygen_effect and params.pO2 is not None
    osmk_version = None
    if is_osmk:
        has_2021 = all(x is not None for x in (params.zR, params.gamma, params.Rm))
        has_2023 = all(x is not None for x in (params.f_rd_max, params.f_z0_max, params.Rmax))

        if has_2021 and not has_2023:
            osmk_version = "2021"
        elif has_2023 and not has_2021:
            osmk_version = "2023"
        elif has_2021 and has_2023:
            raise ValueError("Inconsistent OSMK input: cannot provide both 2021 and 2023 parameter sets.")
        else:
            raise ValueError("OSMK model requested but required parameters are missing.")

    mktable = params.mktable

    if model == "stochastic" and not mktable.params.use_stochastic_model:
        raise ValueError("Stochastic output requested but MKTable was computed in classic mode.")
    if model == "mcf" and not mktable.params.use_mcf_model:
        raise ValueError("MCF-MKM output requested but MKTable was not computed in MCF mode.")
    if mktable.params.use_mcf_model and model != "mcf":
        raise ValueError("An MCF-mode MKTable can only be used with model='mcf'.")

    # z0 is needed only for MKM/SMK recomputation. Keep it local: SFTable.compute
    # must not mutate the MKTable configuration.
    z0 = None
    if model != "mcf":
        z0 = mktable.params.z0 or SpecificEnergy.compute_saturation_parameter(
            domain_radius=mktable.params.domain_radius,
            nucleus_radius=mktable.params.nucleus_radius,
            beta0=params.beta0,
        )
        z0 = round(z0, 2)

    full_ion_name = mktable.sp_table_set._map_to_fullname(ion)

    ion_data = None
    if mktable.table and full_ion_name in mktable.table:
        ion_data = mktable.table[full_ion_name].get("data")

    sp = mktable.sp_table_set.get(full_ion_name)

    def _compute(
        energy_val: float,
        let_val: float,
        *,
        domain_radius: Optional[float] = None,
        z0_override: Optional[float] = None,
    ) -> dict:
        params_dict = _build_worker_params(
            mktable,
            integration_method="trapz",
            domain_radius=domain_radius,
            z0=z0_override,
        )
        return _compute_for_energy_let_pair(
            params_dict,
            energy=energy_val,
            let=let_val,
            atomic_number=sp.atomic_number,
        )

    if energy is None and let is None:
        raise ValueError("At least one of 'energy' or 'let' must be specified.")

    # Each request is (energy, LET, optional matching MKTable row).
    requests = []

    if force_recompute or ion_data is None:
        if energy is not None and let is None:
            inferred_let = sp.interpolate(energy=energy)[0]
            requests = [(energy, inferred_let, None)]
        elif let is not None and energy is None:
            energy_map = sp.interpolate(let=let)
            requests = [(e, let, None) for e in energy_map[let]]
        else:
            requests = [(energy, let, None)]
    else:
        if energy is not None and let is not None:
            matches = _find_precomputed_rows(ion_data, energy=energy, let=let)
            row = matches.iloc[0] if not matches.empty else None
            requests = [(energy, let, row)]
        elif energy is not None:
            matches = _find_precomputed_rows(ion_data, energy=energy)
            if not matches.empty:
                requests = [
                    (float(row["energy"]), float(row["let"]), row)
                    for _, row in matches.iterrows()
                ]
            else:
                inferred_let = sp.interpolate(energy=energy)[0]
                requests = [(energy, inferred_let, None)]
        elif let is not None:
            matches = _find_precomputed_rows(ion_data, let=let)
            if not matches.empty:
                requests = [
                    (float(row["energy"]), float(row["let"]), row)
                    for _, row in matches.iterrows()
                ]
            else:
                energy_map = sp.interpolate(let=let)
                requests = [(e, let, None) for e in energy_map[let]]

    results = []
    for E, L, precomputed_row in requests:
        result = None
        calc_info = "computed"

        if precomputed_row is not None:
            result = _precomputed_result(precomputed_row, model)
            if result is not None:
                calc_info = "precomputed"

        if result is None:
            result = _compute(E, L, z0_override=z0)

        alpha0 = params.alpha0
        beta0 = params.beta0
        dose_grid = params.dose_grid

        if model == "mcf":
            c_bar = result["c_bar"]
            z_bar_c = result["z_bar_c"]

            alpha_MCF, beta_MCF = compute_mcf_lq_coefficients(
                c_bar=c_bar,
                z_bar_c=z_bar_c,
                alpha0=alpha0,
                beta0=beta0,
            )
            sf_curve = np.exp(-alpha_MCF * dose_grid - beta_MCF * dose_grid ** 2)

        else:
            z_bar_star_domain = result["z_bar_star_domain"]
            z_bar_domain = result.get("z_bar_domain")
            z_bar_nucleus = result.get("z_bar_nucleus")

            alpha_MKM, beta_MKM = compute_mkm_lq_coefficients(
                z_bar_star_domain=z_bar_star_domain,
                alpha0=alpha0,
                beta0=beta0,
            )

            if model == "classic":
                sf_curve = np.exp(-alpha_MKM * dose_grid - beta_MKM * dose_grid ** 2)

            elif model == "stochastic":
                if is_osmk:
                    if osmk_version == "2021":
                        R = compute_osmk2021_radioresistance(
                            z_bar_domain=z_bar_domain,
                            K=params.K,
                            pO2=params.pO2,
                            zR=params.zR,
                            gamma=params.gamma,
                            Rm=params.Rm,
                        )
                        f_rd = f_z0 = None
                    else:
                        R, f_rd, f_z0 = compute_osmk2023_radioresistance(
                            K=params.K,
                            pO2=params.pO2,
                            Rmax=params.Rmax,
                            f_rd_max=params.f_rd_max,
                            f_z0_max=params.f_z0_max,
                        )

                    if f_rd is not None and f_z0 is not None:
                        rd_OSMK, z0_OSMK = compute_osmk2023_effective_parameters(
                            mktable.params.domain_radius, z0, f_rd, f_z0
                        )

                        # OSMK 2023 changes the effective domain geometry. Recompute
                        # with worker overrides instead of mutating MKTable.params.
                        result_osmk = _compute(
                            E,
                            L,
                            domain_radius=rd_OSMK,
                            z0_override=z0_OSMK,
                        )
                        z_bar_star_domain = result_osmk["z_bar_star_domain"]
                        z_bar_domain = result_osmk["z_bar_domain"]
                        z_bar_nucleus = result_osmk["z_bar_nucleus"]
                        calc_info = "computed"

                    alpha_SMK, beta_SMK = compute_osmk_lq_coefficients(
                        z_bar_star_domain=z_bar_star_domain,
                        z_bar_domain=z_bar_domain,
                        R=R,
                        alphaL=params.alphaL,
                        alphaS=params.alphaS,
                        beta0=params.beta0,
                    )

                else:
                    alpha_SMK, beta_SMK = compute_smk_lq_coefficients(
                        z_bar_star_domain=z_bar_star_domain,
                        z_bar_domain=z_bar_domain,
                        alpha0=alpha0,
                        beta0=beta0,
                    )

                gamma_SMK = compute_smk_gamma(
                    alpha_smk=alpha_SMK,
                    beta_smk=beta_SMK,
                    z_bar_nucleus=z_bar_nucleus,
                    dose=dose_grid,
                )
                exponent = alpha_SMK * dose_grid + beta_SMK * dose_grid ** 2
                correction_factor = 1 + gamma_SMK * dose_grid
                sf_curve = np.exp(-exponent) * correction_factor

        results.append({
            "params": {
                "ion": full_ion_name,
                "energy": float(E),
                "let": float(L),
                "model": model,
                "osmk_version": osmk_version if is_osmk else None,
            },
            "calculation_info": calc_info,
            "data": pd.DataFrame({
                "dose": dose_grid,
                "survival_fraction": sf_curve,
            }),
        })

    self.table = results


SFTable.compute = compute
