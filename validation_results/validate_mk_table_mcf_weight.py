import locale
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Access local project modules before any installed pyMKM package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pymkm.biology.mcf_model import (
    compute_mcf_correction_factor,
    compute_scaled_nuclear_specific_energy,
)
from validation_results.validation_utils.loader import load_validation_file
from validation_results.validation_utils.metrics import semi_log_error_metrics


locale.setlocale(locale.LC_ALL, "")
csv_sep = ";" if locale.getlocale()[0] == "Italian_Italy" else ","

# SI constants used only for the unit conversion
MEV_TO_J = 1.602176634e-13
CM_TO_M = 1.0e-2
UM_TO_M = 1.0e-6
RHO_WATER_KG_M3 = 1000.0

QUANTITIES = {
    "alpha": {
        "panel": "(A)",
        "title": r"$\alpha(y_{\mathrm{ATS}})$",
        "ylabel": r"$\alpha(y_{\mathrm{ATS}})$ [Gy$^{-1}$]",
        "yscale": "linear",
    },
    "r_RBE_alpha": {
        "panel": "(B)",
        "title": r"$rRBE_{\alpha}(y_{\mathrm{ATS}})$",
        "ylabel": r"$rRBE_{\alpha}(y_{\mathrm{ATS}})$ [-]",
        "yscale": "linear",
    },
    "e_RBE_alpha": {
        "panel": "(C)",
        "title": r"$e(y_{\mathrm{ATS}})$",
        "ylabel": r"$e(y_{\mathrm{ATS}})$ [MeV/cm]",
        "yscale": "log",
    },
    "c": {
        "panel": "(D)",
        "title": r"$c(y_{\mathrm{ATS}})$",
        "ylabel": r"$c(y_{\mathrm{ATS}})$ [-]",
        "yscale": "linear",
    },
}

PARAMETER_KEYS = (
    "Cell_Type",
    "Domain_Radius_um",
    "Nucleus_Radius_um",
    "Alpha0_Gy-1",
    "Beta0_Gy-2",
    "AlphaRef_Gy-1",
    "BetaRef_Gy-2",
)


def _ats_equivalent_lineal_energy_to_specific_energy(
    y_ats_mev_cm: np.ndarray,
    radius_um: float,
) -> np.ndarray:
    """
    Convert ATS-equivalent lineal energy to single-event specific energy.

    The one-dimensional MCF weighting functions are represented on the
    ATS-equivalent variable

        y_ATS = rho * pi * r^2 * z

    so that

        z = y_ATS / (rho * pi * r^2).

    Here y_ATS is supplied in MeV/cm, radius in micrometres, rho is water
    density, and z is returned in Gy.
    """
    y_ats_mev_cm = np.asarray(y_ats_mev_cm, dtype=float)
    if radius_um <= 0:
        raise ValueError("radius_um must be strictly positive.")

    energy_per_length_j_m = y_ats_mev_cm * MEV_TO_J / CM_TO_M
    radius_m = radius_um * UM_TO_M
    mass_per_length_kg_m = RHO_WATER_KG_M3 * np.pi * radius_m ** 2

    return energy_per_length_j_m / mass_per_length_kg_m


def _compute_ats_weighting_functions(
    y_ats_mev_cm: np.ndarray,
    domain_radius_um: float,
    nucleus_radius_um: float,
    alpha0: float,
    beta0: float,
    alpha_ref: float,
) -> dict:
    """
    Compute canonical MCF-MKM weighting functions in ATS-equivalent form.

    This representation corresponds to the Parisi-consistent ``scaled``
    nucleus treatment. The ``integrated`` treatment cannot be represented as
    a single-valued function of one y_ATS variable because z_d and z_n are then
    calculated independently.
    """
    y_ats_mev_cm = np.asarray(y_ats_mev_cm, dtype=float)

    z_domain = _ats_equivalent_lineal_energy_to_specific_energy(
        y_ats_mev_cm,
        domain_radius_um,
    )
    z_nucleus = compute_scaled_nuclear_specific_energy(
        z_domain=z_domain,
        domain_radius=domain_radius_um,
        nucleus_radius=nucleus_radius_um,
    )
    c = compute_mcf_correction_factor(
        z_domain=z_domain,
        z_nucleus=z_nucleus,
        alpha0=alpha0,
        beta0=beta0,
    )

    alpha_weight = (alpha0 + beta0 * z_domain) * c
    r_rbe_alpha = alpha_weight / alpha_ref
    e_rbe_alpha = y_ats_mev_cm * r_rbe_alpha

    return {
        "alpha": alpha_weight,
        "r_RBE_alpha": r_rbe_alpha,
        "e_RBE_alpha": e_rbe_alpha,
        "c": c,
    }


def _load_cell_reference(cell_dir: Path):
    reference = {}

    for quantity in QUANTITIES:
        quantity_dir = cell_dir / quantity
        files = sorted(quantity_dir.glob("*.txt"))

        if len(files) != 1:
            raise ValueError(
                f"Expected exactly one reference file in '{quantity_dir}', "
                f"found {len(files)}."
            )

        metadata, df_ref = load_validation_file(files[0])
        reference[quantity] = {
            "file": files[0],
            "metadata": metadata,
            "data": df_ref,
        }

    # All four digitized curves must belong to the same biological parameter set.
    first_metadata = reference["alpha"]["metadata"]
    for quantity, item in reference.items():
        metadata = item["metadata"]
        for key in PARAMETER_KEYS:
            if metadata.get(key) != first_metadata.get(key):
                raise ValueError(
                    f"Inconsistent metadata for '{key}' in "
                    f"{item['file'].name} ({quantity})."
                )

    return first_metadata, reference


def validate_mk_table_mcf_weight():
    """
    Validate the MCF-MKM biological weighting functions from Parisi et al. (2022)
    using their ATS-equivalent one-dimensional representation.

    The digitized abscissa is interpreted as y_ATS and is stored in MeV/cm.
    This validation therefore tests the canonical/scaled MCF biological kernel,
    not an ion-specific track structure or the alternative integrated nucleus mode.
    """
    base_dir = Path(__file__).resolve().parent / "mk_table_mcf_weight"
    ref_dir = base_dir / "reference_data"
    fig_dir = base_dir / "figures"
    metrics_dir = base_dir / "metrics"
    fig_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    cell_dirs = sorted(path for path in ref_dir.iterdir() if path.is_dir())
    if not cell_dirs:
        raise FileNotFoundError(f"No cell-line folders found in '{ref_dir}'.")

    metric_records = []

    for cell_dir in cell_dirs:
        metadata, reference = _load_cell_reference(cell_dir)

        cell_type = metadata["Cell_Type"]
        domain_radius = float(metadata["Domain_Radius_um"])
        nucleus_radius = float(metadata["Nucleus_Radius_um"])
        alpha0 = float(metadata["Alpha0_Gy-1"])
        beta0 = float(metadata["Beta0_Gy-2"])
        alpha_ref = float(metadata["AlphaRef_Gy-1"])
        beta_ref = float(metadata["BetaRef_Gy-2"])

        all_x = np.concatenate(
            [np.asarray(reference[q]["data"]["x"], dtype=float) for q in QUANTITIES]
        )
        positive_x = all_x[np.isfinite(all_x) & (all_x > 0)]
        if positive_x.size == 0:
            raise ValueError(f"No positive y_ATS values available for {cell_type}.")

        y_min = positive_x.min()
        y_max = positive_x.max()
        y_dense = np.logspace(
            np.log10(y_min) - 0.03,
            np.log10(y_max) + 0.03,
            1200,
        )

        model_dense = _compute_ats_weighting_functions(
            y_ats_mev_cm=y_dense,
            domain_radius_um=domain_radius,
            nucleus_radius_um=nucleus_radius,
            alpha0=alpha0,
            beta0=beta0,
            alpha_ref=alpha_ref,
        )

        fig, axs = plt.subplots(2, 3, figsize=(18, 10), squeeze=False)
        plot_axes = {
            "alpha": axs[0, 0],
            "r_RBE_alpha": axs[0, 1],
            "e_RBE_alpha": axs[0, 2],
            "c": axs[1, 0],
        }

        # Keep the middle lower panel empty to retain the same 2x3 validation layout.
        axs[1, 1].axis("off")

        for quantity, ax in plot_axes.items():
            df_ref = reference[quantity]["data"]
            x_ref = np.asarray(df_ref["x"], dtype=float)
            y_ref = np.asarray(df_ref["y"], dtype=float)
            y_model = model_dense[quantity]

            ax.plot(
                y_dense,
                y_model,
                linewidth=3,
                linestyle='--',
                color='black',
                label="pyMKM MCF-MKM (ATS, scaled)",
            )
            ax.plot(
                x_ref,
                y_ref,
                alpha=0.4, 
                linewidth=6, 
                color='gray',
                label="Parisi et al. (2022)",
            )

            metrics = semi_log_error_metrics(
                x_ref=x_ref,
                y_ref=y_ref,
                x_model=y_dense,
                y_model=y_model,
            )

            metric_records.append(
                {
                    "cell_line": cell_type,
                    "quantity": quantity,
                    "mcf_nucleus_mode": "scaled",
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
                }
            )

            print(
                f"{cell_type} | {quantity} | "
                f"MAE={metrics['mean_abs_error']:.4g}, "
                f"RMS={metrics['rms_error']:.4g}, "
                f"Max={metrics['max_abs_error']:.4g}, "
                f"SMAPE={metrics['smape_percent']:.2f}%"
            )

            spec = QUANTITIES[quantity]
            ax.set_xscale("log")
            ax.set_yscale(spec["yscale"])
            ax.set_xlabel(
                r"ATS-equivalent lineal energy, $y_{\mathrm{ATS}}$ [MeV/cm]"
            )
            ax.set_ylabel(spec["ylabel"])
            ax.set_title(f"{spec['panel']} {spec['title']}", fontsize=14)
            ax.grid(True, which="both", linestyle="--", alpha=0.3)
            ax.legend(fontsize=10)

        info_ax = axs[1, 2]
        info_ax.axis("off")
        info_text = (
            f"{cell_type}\n"
            f"$\\alpha_0$: {alpha0:.3f} Gy$^{{-1}}$\n"
            f"$\\beta_0$: {beta0:.3f} Gy$^{{-2}}$\n"
            f"$\\alpha_{{ref}}$: {alpha_ref:.3f} Gy$^{{-1}}$\n"
            f"$\\beta_{{ref}}$: {beta_ref:.3f} Gy$^{{-2}}$\n"
            f"$r_d$: {domain_radius:.2f} μm\n"
            f"$R_n$: {nucleus_radius:.2f} μm\n"
            f"MCF nucleus mode: scaled"
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
            f"MCF-MKM biological weighting functions — {cell_type} | "
            "ATS-equivalent representation",
            fontsize=16,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        fig_path = fig_dir / f"{cell_type}_mcf_weight_ATS.png"
        fig.savefig(fig_path, dpi=300)
        plt.close(fig)

    metrics_path = metrics_dir / "mk_table_mcf_weight_metrics.csv"
    pd.DataFrame(metric_records).to_csv(
        metrics_path,
        sep=csv_sep,
        index=False,
    )
    print(f"\nSaved validation metrics to: {metrics_path}")

    return pd.DataFrame(metric_records)


if __name__ == "__main__":
    validate_mk_table_mcf_weight()
