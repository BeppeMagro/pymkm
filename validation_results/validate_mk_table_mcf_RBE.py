"""
Validate MCF-MKM RBE quantities for V79 cells against Parisi et al. data.

The validation uses the public pyMKM API throughout:

    StoppingPowerTableSet -> MKTable(MCF) -> c_bar, z_bar_c
        -> compute_mcf_lq_coefficients -> RBE

The ``scaled`` nucleus mode is the pyMKM default because it preserves the
original MCF domain-to-nucleus scaling used by Parisi et al. The ``integrated``
mode, which evaluates nuclear specific energy directly over the nucleus geometry
within the amorphous-track formulation, is retained as an alternative.
"""

from pathlib import Path
import locale
import sys
from typing import Dict, Iterable, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Access local pyMKM when this file is executed directly.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from validation_results.validation_utils.inverse_dose import compute_lq_dose_from_survival
from validation_results.validation_utils.loader import load_validation_file
from validation_results.validation_utils.metrics import semi_log_error_metrics
from pymkm.biology.mcf_model import compute_mcf_lq_coefficients
from pymkm.io.table_set import StoppingPowerTableSet
from pymkm.mktable.core import MKTable, MKTableParameters


locale.setlocale(locale.LC_ALL, "")
csv_sep = ";" if locale.getlocale()[0] == "Italian_Italy" else ","

ENDPOINTS = {
    "RBE_alpha": {
        "column": "rbe_alpha",
        "label": r"$\mathrm{RBE}_{\alpha}$",
        "survival": None,
    },
    "RBE_beta": {
        "column": "rbe_beta",
        "label": r"$\mathrm{RBE}_{\beta}$",
        "survival": None,
    },
    "RBE_0.5": {
        "column": "rbe_0.5",
        "label": r"$\mathrm{RBE}_{50}$",
        "survival": 0.5,
    },
    "RBE_0.1": {
        "column": "rbe_0.1",
        "label": r"$\mathrm{RBE}_{10}$",
        "survival": 0.1,
    },
    "RBE_0.01": {
        "column": "rbe_0.01",
        "label": r"$\mathrm{RBE}_{1}$",
        "survival": 0.01,
    },
}

_METADATA_KEYS = (
    "Reference",
    "Model_Name",
    "Core_Radius_Type",
    "Cell_Type",
    "Atomic_Number",
    "Domain_Radius_um",
    "Nucleus_Radius_um",
    "Alpha0_Gy-1",
    "Beta0_Gy-2",
    "AlphaRef_Gy-1",
    "BetaRef_Gy-2",
)


def _load_reference_sets(ref_root: Path) -> Dict[str, Dict[int, Tuple[dict, pd.DataFrame, Path]]]:
    """Load all endpoint reference files and index them by endpoint and atomic number."""
    reference_sets = {}
    for endpoint in ENDPOINTS:
        endpoint_dir = ref_root / endpoint
        files = sorted(endpoint_dir.glob("*.txt"))
        if not files:
            raise FileNotFoundError(f"No reference files found in: {endpoint_dir}")

        by_z = {}
        for path in files:
            metadata, df = load_validation_file(path)
            z = int(metadata["Atomic_Number"])
            if z in by_z:
                raise ValueError(f"Duplicate {endpoint} reference dataset for Z={z}.")
            by_z[z] = (metadata, df, path)
        reference_sets[endpoint] = by_z

    z_sets = [set(by_z) for by_z in reference_sets.values()]
    if any(z_set != z_sets[0] for z_set in z_sets[1:]):
        raise ValueError("MCF-MKM reference endpoint folders do not contain the same ion set.")

    # All endpoints for a given ion must describe exactly the same biological/model setup.
    for z in sorted(z_sets[0]):
        signatures = []
        for endpoint in ENDPOINTS:
            metadata = reference_sets[endpoint][z][0]
            signatures.append(tuple(metadata.get(key) for key in _METADATA_KEYS))
        if any(signature != signatures[0] for signature in signatures[1:]):
            raise ValueError(f"Inconsistent MCF-MKM metadata across endpoints for Z={z}.")

    return reference_sets


def _rbe_at_survival(
    alpha: np.ndarray,
    beta: np.ndarray,
    alpha_ref: float,
    beta_ref: float,
    survival: float,
) -> np.ndarray:
    """Compute exact LQ RBE at a fixed surviving fraction."""
    d_ref = compute_lq_dose_from_survival(alpha_ref, beta_ref, survival)
    d_ion = np.array(
        [compute_lq_dose_from_survival(float(a), float(b), survival) for a, b in zip(alpha, beta)],
        dtype=float,
    )
    return d_ref / d_ion


def _compute_mcf_predictions(
    *,
    source: str,
    z: int,
    metadata: dict,
    nucleus_mode: str,
    parallel: bool,
) -> Tuple[pd.DataFrame, StoppingPowerTableSet]:
    """Compute one complete MCF-MKM table and all derived RBE quantities for one ion."""
    domain_radius = float(metadata["Domain_Radius_um"])
    nucleus_radius = float(metadata["Nucleus_Radius_um"])
    alpha0 = float(metadata["Alpha0_Gy-1"])
    beta0 = float(metadata["Beta0_Gy-2"])
    alpha_ref = float(metadata["AlphaRef_Gy-1"])
    beta_ref = float(metadata["BetaRef_Gy-2"])

    sp_table_set = StoppingPowerTableSet.from_default_source(source).filter_by_ions([z])
    params = MKTableParameters(
        domain_radius=domain_radius,
        nucleus_radius=nucleus_radius,
        alpha0=alpha0,
        beta0=beta0,
        model_name=metadata["Model_Name"],
        core_radius_type=metadata["Core_Radius_Type"],
        use_mcf_model=True,
        mcf_nucleus_mode=nucleus_mode,
    )
    mk_table = MKTable(parameters=params, sp_table_set=sp_table_set)
    mk_table.compute(ions=[z], parallel=parallel)

    df = mk_table.get_table(z).copy()
    coefficients = [
        compute_mcf_lq_coefficients(
            c_bar=float(c_bar),
            z_bar_c=float(z_bar_c),
            alpha0=alpha0,
            beta0=beta0,
        )
        for c_bar, z_bar_c in zip(df["c_bar"], df["z_bar_c"])
    ]
    alpha_mcf = np.array([alpha for alpha, _ in coefficients], dtype=float)
    beta_mcf = np.array([beta for _, beta in coefficients], dtype=float)

    df["alpha_mcf"] = alpha_mcf
    df["beta_mcf"] = beta_mcf
    df["rbe_alpha"] = alpha_mcf / alpha_ref
    # The supplied RBE_beta reference datasets use beta / beta_ref, matching
    # the previous pyMKM validation scripts.
    df["rbe_beta"] = beta_mcf / beta_ref

    for endpoint, config in ENDPOINTS.items():
        survival = config["survival"]
        if survival is not None:
            df[config["column"]] = _rbe_at_survival(
                alpha_mcf,
                beta_mcf,
                alpha_ref,
                beta_ref,
                survival,
            )

    # Reference comparisons use LET as the independent variable.
    df = df.sort_values("let").reset_index(drop=True)
    return df, sp_table_set


def validate_mk_table_mcf_RBE(
    source: str = "mstar_3_12",
    nucleus_modes: Sequence[str] = ("scaled", "integrated"),
    atomic_numbers: Optional[Iterable[int]] = None,
    parallel: bool = True,
):
    """
    Validate V79 MCF-MKM RBE_alpha, RBE_beta, RBE50, RBE10, and RBE1.

    Parameters
    ----------
    source : str
        Stopping-power source used by pyMKM.
    nucleus_modes : sequence of {"integrated", "scaled"}
        MCF nucleus treatments to evaluate. ``scaled`` is the pyMKM
        default and Parisi-consistent implementation; ``integrated`` is retained as the alternative.
    atomic_numbers : iterable of int, optional
        Restrict the validation to selected ions. By default all reference ions
        are evaluated.
    parallel : bool
        Use MKTable multiprocessing.
    """
    allowed_modes = {"scaled", "integrated"}
    nucleus_modes = tuple(nucleus_modes)
    if not nucleus_modes or any(mode not in allowed_modes for mode in nucleus_modes):
        raise ValueError("nucleus_modes must contain 'scaled' and/or 'integrated'.")

    base_dir = Path(__file__).resolve().parent / "mk_table_mcf_RBE"
    ref_root = base_dir / "reference_data" / "V79"
    fig_dir = base_dir / "figures"
    metrics_dir = base_dir / "metrics"
    fig_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    references = _load_reference_sets(ref_root)
    available_z = sorted(references["RBE_alpha"])
    if atomic_numbers is None:
        selected_z = available_z
    else:
        selected_z = sorted(set(int(z) for z in atomic_numbers))
        missing = [z for z in selected_z if z not in available_z]
        if missing:
            raise ValueError(f"No MCF-MKM V79 reference data available for Z={missing}.")

    metric_records = []

    for z in selected_z:
        metadata = references["RBE_alpha"][z][0]
        alpha0 = float(metadata["Alpha0_Gy-1"])
        beta0 = float(metadata["Beta0_Gy-2"])
        alpha_ref = float(metadata["AlphaRef_Gy-1"])
        beta_ref = float(metadata["BetaRef_Gy-2"])
        domain_radius = float(metadata["Domain_Radius_um"])
        nucleus_radius = float(metadata["Nucleus_Radius_um"])

        predictions = {}
        sp_table_set = None
        for nucleus_mode in nucleus_modes:
            print(f"\nMCF-MKM validation: Z={z}, nucleus_mode={nucleus_mode}, source={source}")
            df_model, sp_table_set = _compute_mcf_predictions(
                source=source,
                z=z,
                metadata=metadata,
                nucleus_mode=nucleus_mode,
                parallel=parallel,
            )
            predictions[nucleus_mode] = df_model

            # Keep the complete computed table for reproducibility and later analysis.
            prediction_path = metrics_dir / f"mcf_RBE_predictions_Z{z}_{nucleus_mode}_{source}.csv"
            df_model.to_csv(prediction_path, sep=csv_sep, index=False)

            for endpoint, config in ENDPOINTS.items():
                _, df_ref, path = references[endpoint][z]
                x_ref = df_ref["x"].to_numpy(dtype=float)
                y_ref = df_ref["y"].to_numpy(dtype=float)
                x_model = df_model["let"].to_numpy(dtype=float)
                y_model = df_model[config["column"]].to_numpy(dtype=float)

                metrics = semi_log_error_metrics(x_ref, y_ref, x_model, y_model)
                metric_records.append({
                    "reference_file": path.name,
                    "cell_type": metadata["Cell_Type"],
                    "Z": z,
                    "endpoint": endpoint,
                    "nucleus_mode": nucleus_mode,
                    "primary_mode": nucleus_mode == "scaled",
                    "source": source,
                    "model": metadata["Model_Name"],
                    "core_radius_type": metadata["Core_Radius_Type"],
                    "domain_radius_um": domain_radius,
                    "nucleus_radius_um": nucleus_radius,
                    "alpha0_Gy-1": alpha0,
                    "beta0_Gy-2": beta0,
                    "alpha_ref_Gy-1": alpha_ref,
                    "beta_ref_Gy-2": beta_ref,
                    "MeanAbsError": metrics["mean_abs_error"],
                    "RMS_Error": metrics["rms_error"],
                    "MaxAbsError": metrics["max_abs_error"],
                    "SMAPE_percent": metrics["smape_percent"],
                })
                print(
                    f"  {endpoint}: SMAPE={metrics['smape_percent']:.2f}% | "
                    f"mean={metrics['mean_abs_error']:.3f} | max={metrics['max_abs_error']:.3f}"
                )

        # One summary figure per ion, with the sixth panel reserved for
        # the biological/model parameters, consistently with the other validations.
        fig, axes = plt.subplots(2, 3, figsize=(18, 10), squeeze=False)
        axes = axes.flatten()
        color = sp_table_set.get(z).color
        line_styles = {"scaled": "-", "integrated": ":"}

        for ax, (endpoint, config) in zip(axes, ENDPOINTS.items()):
            _, df_ref, _ = references[endpoint][z]
            for nucleus_mode in nucleus_modes:
                df_model = predictions[nucleus_mode]
                mode_label = (
                    "pyMKM MCF (scaled, default)"
                    if nucleus_mode == "scaled"
                    else "pyMKM MCF (integrated)"
                )
                ax.semilogx(
                    df_model["let"],
                    df_model[config["column"]],
                    line_styles[nucleus_mode],
                    linewidth=4 if nucleus_mode == "scaled" else 3,
                    alpha=0.60 if nucleus_mode == "scaled" else 0.75,
                    color=color,
                    label=mode_label,
                )
            ax.semilogx(
                df_ref["x"],
                df_ref["y"],
                marker="o",
                linestyle="None",
                markersize=8,
                markeredgewidth=0.8,
                markeredgecolor="black",
                alpha=0.7,
                color=color,
                label=metadata.get("Reference", "Reference"),
            )
            ax.set_xlabel("LET [MeV/cm]")
            ax.set_ylabel(config["label"])
            ax.set_title(endpoint.replace("_", " "), fontsize=14)
            ax.grid(True, which="both", linestyle="--", alpha=0.1)
            ax.legend(loc="best", fontsize=10)

        # Match the information-box style used throughout the existing
        # validation suite, while exploiting the otherwise empty sixth panel.
        info_ax = axes[-1]
        info_ax.axis("off")

        info_text = (
            f"{metadata['Cell_Type']}\n"
            f"$\\alpha_0$: {alpha0:.3f} Gy$^{{-1}}$\n"
            f"$\\beta_0$: {beta0:.3f} Gy$^{{-2}}$\n"
            f"$r_d$: {domain_radius:.2f} μm\n"
            f"$R_n$: {nucleus_radius:.2f} μm\n"
            f"$\\alpha_{{ref}}$: {alpha_ref:.3f} Gy$^{{-1}}$\n"
            f"$\\beta_{{ref}}$: {beta_ref:.3f} Gy$^{{-2}}$"
        )
        info_ax.text(
            0.05,
            0.95,
            info_text,
            transform=info_ax.transAxes,
            fontsize=14,
            verticalalignment="top",
            horizontalalignment="left",
            bbox=dict(
                facecolor="white",
                alpha=0.85,
                edgecolor="black",
                boxstyle="round",
            ),
        )

        fig.suptitle(
            f"Source: {source}, Track model: {metadata['Model_Name']} "
            f"(Core: {metadata['Core_Radius_Type']})\n"
            f"MCF-MKM RBE validation — {metadata['Cell_Type']}, Z = {z}",
            fontsize=16,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        fig.savefig(fig_dir / f"{metadata['Cell_Type']}_Z{z}_mcf_RBE_{source}.png", dpi=300)
        plt.close(fig)

    metrics_path = metrics_dir / f"mk_table_mcf_RBE_metrics_{source}.csv"
    pd.DataFrame(metric_records).to_csv(metrics_path, sep=csv_sep, index=False)
    print(f"\nSaved MCF-MKM validation metrics to: {metrics_path}")
    return pd.DataFrame(metric_records)


if __name__ == "__main__":
    validate_mk_table_mcf_RBE(source="mstar_3_12")
