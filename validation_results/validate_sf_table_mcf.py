import matplotlib.pyplot as plt
from pathlib import Path
import sys
from collections import defaultdict
from typing import Sequence

# Ensure the local development checkout takes precedence over any installed pyMKM version.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from validation_results.validation_utils.layout import choose_horizontal_subplot_layout
from validation_results.validation_utils.loader import load_validation_file
from validation_results.validation_utils.metrics import log_linear_error_metrics

from pymkm.mktable.core import MKTableParameters, MKTable
from pymkm.sftable.core import SFTableParameters, SFTable
from pymkm.io.table_set import StoppingPowerTableSet

import locale
locale.setlocale(locale.LC_ALL, '')
csv_sep = ';' if locale.getlocale()[0] == 'Italian_Italy' else ','

import numpy as np
import pandas as pd


_VALID_MCF_NUCLEUS_MODES = ("scaled", "integrated")
_MODE_LINESTYLES = {
    "scaled": "-",
    "integrated": ":",
}


def validate_sf_table_mcf(
    source: str = "mstar_3_12",
    nucleus_modes: Sequence[str] = ("scaled", "integrated"),
):
    """
    Validate MCF-MKM survival fraction curves computed by SFTable against
    published reference curves.

    Both available MCF nucleus-specific-energy prescriptions are evaluated by
    default. ``scaled`` is the package default and Parisi-consistent implementation;
    ``integrated`` is retained as the alternative amorphous-track treatment.

    Parameters
    ----------
    source : str
        Stopping-power source used by pyMKM.
    nucleus_modes : sequence of {"integrated", "scaled"}
        MCF-MKM nucleus-specific-energy implementations to evaluate.
    """
    nucleus_modes = tuple(nucleus_modes)
    if not nucleus_modes:
        raise ValueError("nucleus_modes must contain at least one MCF nucleus mode.")
    invalid_modes = [mode for mode in nucleus_modes if mode not in _VALID_MCF_NUCLEUS_MODES]
    if invalid_modes:
        raise ValueError(
            "nucleus_modes can contain only 'integrated' and 'scaled'. "
            f"Invalid values: {invalid_modes}."
        )

    base_dir = Path(__file__).resolve().parent / "sf_table_mcf"
    ref_dir = base_dir / "reference_data"
    fig_dir = base_dir / "figures"
    metrics_dir = base_dir / "metrics"
    fig_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    error_records = []

    for cell_folder in sorted(path for path in ref_dir.iterdir() if path.is_dir()):
        cell_line = cell_folder.name
        grouped_by_Z = defaultdict(list)

        for txt_file in sorted(cell_folder.glob("*.txt")):
            metadata, df_ref = load_validation_file(txt_file)
            Z = int(metadata["Atomic_Number"])
            let_val = float(metadata["LET_MeV_cm"])
            grouped_by_Z[Z].append((let_val, txt_file, metadata, df_ref))

        for Z in sorted(grouped_by_Z):
            entries = sorted(grouped_by_Z[Z], key=lambda item: item[0])
            layout_list = choose_horizontal_subplot_layout(len(entries), max_cols_per_fig=3)
            entry_idx = 0

            _, _, first_metadata, _ = entries[0]
            cell_type = first_metadata["Cell_Type"]
            domain_radius = float(first_metadata["Domain_Radius_um"])
            nucleus_radius = float(first_metadata["Nucleus_Radius_um"])
            alpha0 = float(first_metadata["Alpha0_Gy-1"])
            beta0 = float(first_metadata["Beta0_Gy-2"])
            alpha_ref = float(first_metadata["AlphaRef_Gy-1"])
            beta_ref = float(first_metadata["BetaRef_Gy-2"])
            model_name = first_metadata["Model_Name"]
            core_type = first_metadata["Core_Radius_Type"]

            sp_table_set = StoppingPowerTableSet.from_default_source(source).filter_by_ions([Z])
            ion_info = sp_table_set.get(Z)
            color = ion_info.color
            ion_symbol = ion_info.ion_symbol

            mk_tables = {}
            for nucleus_mode in nucleus_modes:
                mk_params = MKTableParameters(
                    domain_radius=domain_radius,
                    nucleus_radius=nucleus_radius,
                    alpha0=alpha0,
                    beta0=beta0,
                    model_name=model_name,
                    core_radius_type=core_type,
                    use_mcf_model=True,
                    mcf_nucleus_mode=nucleus_mode,
                )
                mk_tables[nucleus_mode] = MKTable(
                    parameters=mk_params,
                    sp_table_set=sp_table_set,
                )

            for fig_idx, (n_rows, n_cols) in enumerate(layout_list):
                fig, axs = plt.subplots(
                    n_rows,
                    n_cols,
                    figsize=(6 * n_cols, 5),
                    squeeze=False,
                )
                axs = axs[0]

                for ax_idx, ax in enumerate(axs):
                    if entry_idx >= len(entries):
                        ax.axis("off")
                        continue

                    let_val, _, metadata, df_ref = entries[entry_idx]
                    entry_idx += 1

                    # Guard against mixing incompatible biological/model parameters
                    # in the same cell-line / ion figure.
                    current = {
                        "Domain_Radius_um": domain_radius,
                        "Nucleus_Radius_um": nucleus_radius,
                        "Alpha0_Gy-1": alpha0,
                        "Beta0_Gy-2": beta0,
                        "AlphaRef_Gy-1": alpha_ref,
                        "BetaRef_Gy-2": beta_ref,
                    }
                    for key, expected in current.items():
                        if not np.isclose(float(metadata[key]), expected, atol=1e-12, rtol=0.0):
                            raise ValueError(
                                f"Inconsistent {key} in {cell_line}, Z={Z}: "
                                f"expected {expected}, found {metadata[key]}."
                            )

                    dose_grid = df_ref["x"].to_numpy(dtype=float)
                    ref_survival = df_ref["y"].to_numpy(dtype=float)
                    ref_label = metadata.get("Reference", "Reference")
                    ax.plot(
                        dose_grid,
                        ref_survival,
                        "--",
                        label=ref_label,
                        linewidth=3,
                        color=color,
                    )

                    for nucleus_mode in nucleus_modes:
                        mk_table = mk_tables[nucleus_mode]
                        sf_params = SFTableParameters(
                            mktable=mk_table,
                            alpha0=alpha0,
                            beta0=beta0,
                            dose_grid=dose_grid,
                        )
                        sf_table = SFTable(parameters=sf_params)
                        sf_table.compute(
                            ion=Z,
                            let=let_val,
                            force_recompute=True,
                            model="mcf",
                        )

                        primary_result = None
                        for result_idx, result in enumerate(sf_table.table):
                            params = result["params"]
                            mode_label = (
                                "Scaled, default" if nucleus_mode == "scaled" else "Integrated"
                            )
                            label = (
                                f"{ion_symbol} (pyMKM MCF-MKM, {mode_label}, "
                                f"E = {params['energy']:.1f} MeV/u)"
                            )
                            ax.plot(
                                result["data"]["dose"],
                                result["data"]["survival_fraction"],
                                _MODE_LINESTYLES[nucleus_mode],
                                label=label,
                                linewidth=6 if nucleus_mode == "scaled" else 5,
                                alpha=0.45 if result_idx == 0 else 0.25,
                                color=color,
                            )
                            if primary_result is None:
                                primary_result = result

                        if primary_result is None:
                            raise RuntimeError(
                                f"SFTable returned no MCF-MKM result for {cell_line}, "
                                f"Z={Z}, LET={let_val}, nucleus_mode={nucleus_mode}."
                            )

                        try:
                            metrics = log_linear_error_metrics(
                                x_ref=dose_grid,
                                y_ref=ref_survival,
                                x_model=primary_result["data"]["dose"].to_numpy(dtype=float),
                                y_model=primary_result["data"]["survival_fraction"].to_numpy(dtype=float),
                            )
                            print(
                                f"{cell_line}, Z = {Z}, LET = {let_val:.1f}, "
                                f"MCF nucleus = {nucleus_mode} | "
                                f"SMAPE_log = {metrics['smape_log']:.2f}%, "
                                f"MeanLogError = {metrics['mean_log_error']:.3f}"
                            )
                        except Exception as exc:
                            print(
                                f"{cell_line}, Z = {Z}, LET = {let_val:.1f}, "
                                f"MCF nucleus = {nucleus_mode} | "
                                f"Error during comparison: {type(exc).__name__} - {exc}"
                            )
                            metrics = {
                                key: float("nan")
                                for key in (
                                    "mean_log_error",
                                    "rms_log_error",
                                    "max_log_error",
                                    "smape_log",
                                )
                            }

                        error_records.append({
                            "cell_line": cell_line,
                            "Z": Z,
                            "LET_MeV_cm": let_val,
                            "model": model_name,
                            "core_radius_type": core_type,
                            "mcf_nucleus_mode": nucleus_mode,
                            "primary_mode": nucleus_mode == "scaled",
                            "domain_radius_um": domain_radius,
                            "nucleus_radius_um": nucleus_radius,
                            "alpha0_Gy-1": alpha0,
                            "beta0_Gy-2": beta0,
                            "alpha_ref_Gy-1": alpha_ref,
                            "beta_ref_Gy-2": beta_ref,
                            "energy_MeV_u": primary_result["params"]["energy"],
                            "MeanLogError": metrics["mean_log_error"],
                            "RMSLogError": metrics["rms_log_error"],
                            "MaxLogError": metrics["max_log_error"],
                            "SMAPE_log_percent": metrics["smape_log"],
                        })

                    ax.set_xlim(0, 10)
                    ax.set_ylim(1e-4, 1)
                    ax.set_yscale("log")

                    x_label, y_label = map(
                        str.strip,
                        metadata.get(
                            "Data_Units", "Dose [Gy], Survival fraction"
                        ).split(","),
                    )
                    ax.set_xlabel(x_label)
                    ax.set_ylabel(y_label)
                    ax.set_title(f"LET = {let_val:.1f} MeV/cm", fontsize=14)

                    if ax_idx == 0:
                        info_text = (
                            f"{cell_type}\n"
                            f"$\\alpha_0$: {alpha0:.3f} Gy$^{{-1}}$\n"
                            f"$\\beta_0$: {beta0:.3f} Gy$^{{-2}}$\n"
                            f"$\\alpha_{{ref}}$: {alpha_ref:.3f} Gy$^{{-1}}$\n"
                            f"$\\beta_{{ref}}$: {beta_ref:.3f} Gy$^{{-2}}$\n"
                            f"$r_d$: {domain_radius:.2f} μm\n"
                            f"$R_n$: {nucleus_radius:.2f} μm"
                        )
                        ax.text(
                            0.05,
                            0.05,
                            info_text,
                            transform=ax.transAxes,
                            fontsize=14,
                            verticalalignment="bottom",
                            horizontalalignment="left",
                            bbox=dict(
                                facecolor="white",
                                alpha=0.85,
                                edgecolor="black",
                                boxstyle="round",
                            ),
                        )

                    ax.grid(True, which="both", linestyle="--", alpha=0.3)
                    ax.legend(loc="upper right", fontsize=11)

                # Keep the same compact single-line figure title used by the other
                # SFTable validations; the compared MCF modes are identified in the legend.
                title = f"Source: {source} | {model_name} ({core_type} core)"
                fig.suptitle(title, fontsize=13, y=0.98)
                fig.tight_layout(rect=[0, 0, 1, 0.93])

                suffix = f"_{fig_idx + 1}" if len(layout_list) > 1 else ""
                fig_path = fig_dir / f"{cell_line}_Z{Z}_mcf_{source}{suffix}.png"
                fig.savefig(fig_path, dpi=300)
                plt.pause(0.1)

    log_path = metrics_dir / f"sf_table_mcf_metrics_{source}.csv"
    pd.DataFrame(error_records).to_csv(log_path, sep=csv_sep, index=False)
    print(f"\nSaved MCF-MKM survival curve metrics to: {log_path}")


if __name__ == "__main__":
    validate_sf_table_mcf(
        source="mstar_3_12",
        nucleus_modes=("scaled", "integrated"),
    )
    # validate_sf_table_mcf(source="fluka_2020_0")
    # validate_sf_table_mcf(source="geant4_11_3_0")
