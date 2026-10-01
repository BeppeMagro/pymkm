import matplotlib.pyplot as plt
from pathlib import Path
import sys
from collections import defaultdict

# Extend module search path to allow relative imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from validation_results.validation_utils.layout import choose_horizontal_subplot_layout
from validation_results.validation_utils.loader import load_validation_file
from validation_results.validation_utils.metrics import log_linear_error_metrics

from pymkm.mktable.core import MKTableParameters, MKTable
from pymkm.sftable.core import SFTableParameters, SFTable
from pymkm.io.table_set import StoppingPowerTableSet
from pymkm.physics.specific_energy import SpecificEnergy

import locale
locale.setlocale(locale.LC_ALL, '')
csv_sep = ';' if locale.getlocale()[0] == 'Italian_Italy' else ','

import numpy as np
import pandas as pd

def validate_sf_table_classic(source: str = "fluka_2020_0"):
    """
    Validate survival fraction curves computed by SFTable (classic MKM mode)
    against reference datasets for different cell lines and ion species.
    """
    base_dir = Path(__file__).resolve().parent / "sf_table_classic"
    ref_dir = base_dir / "reference_data"
    fig_dir = base_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = base_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    
    error_records = []
    for cell_line in ["HSG", "V79"]:
        folder = ref_dir / cell_line
        grouped_by_Z = defaultdict(list)

        for txt_file in sorted(folder.glob("*.txt")):
            metadata, df_ref = load_validation_file(txt_file)
            Z = int(metadata["Atomic_Number"])
            let_val = float(metadata["LET_MeV_cm"])
            grouped_by_Z[Z].append((let_val, txt_file, metadata, df_ref))

        for Z in sorted(grouped_by_Z):
            entries = grouped_by_Z[Z]
            entries.sort(key=lambda x: x[0])
            layout_list = choose_horizontal_subplot_layout(len(entries), max_cols_per_fig=3)
            entry_idx = 0

            let_val, _, metadata, _ = entries[0]
            cell_type = metadata["Cell_Type"]
            domain_radius = float(metadata["Domain_Radius_um"])
            nucleus_radius = float(metadata["Nucleus_Radius_um"])
            z0 = float(metadata["z0_Gy"])
            beta0 = float(metadata["Beta0_Gy-2"])
            model_name = metadata["Model_Name"]
            core_type = metadata["Core_Radius_Type"]
            atomic_number = int(metadata["Atomic_Number"])

            sp_table_set = StoppingPowerTableSet.from_default_source(source).filter_by_ions([Z])
            mk_params = MKTableParameters(
                domain_radius=domain_radius,
                nucleus_radius=nucleus_radius,
                beta0=beta0,
                model_name=model_name,
                core_radius_type=core_type,
                use_stochastic_model=False
            )
            mk_table = MKTable(parameters=mk_params, sp_table_set=sp_table_set)
            z0_model = round(
                SpecificEnergy.compute_saturation_parameter(
                    domain_radius=domain_radius,
                    nucleus_radius=nucleus_radius,
                    beta0=beta0,
                ),
                2,
            )

            for fig_idx, (n_rows, n_cols) in enumerate(layout_list):
                fig, axs = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5), squeeze=False)
                axs = axs[0]

                for ax_idx, ax in enumerate(axs):
                    if entry_idx >= len(entries):
                        ax.axis("off")
                        continue

                    let_val, txt_file, metadata, df_ref = entries[entry_idx]
                    entry_idx += 1

                    alpha0 = float(metadata["Alpha0_Gy-1"])
                    dose_grid = df_ref["x"].values
                    sf_params = SFTableParameters(mktable=mk_table, alpha0=alpha0, beta0=beta0, dose_grid=dose_grid)
                    sf_table = SFTable(parameters=sf_params)
                    sf_table.compute(ion=atomic_number, let=let_val, force_recompute=True)

                    ref_label = metadata.get("Reference", "Reference") + f" ($z_0$ = {z0:.2f} Gy)"
                    color = sp_table_set.get(atomic_number).color
                    ax.plot(dose_grid, df_ref["y"].values, '--', label=ref_label, linewidth=3, color=color)

                    ion_symbol = sp_table_set.get(atomic_number).ion_symbol
                    for idx, result in enumerate(sf_table.table):
                        model_suffix = "- MKM" if not mk_table.params.use_stochastic_model else "- SMK"
                        params = result["params"]
                        label = f"{ion_symbol} (pyMKM {model_suffix}, E = {params['energy']:.1f} MeV/u)"
                        line_style = '-' if idx == 0 else ':'
                        ax.plot(result["data"]["dose"], result["data"]["survival_fraction"], line_style, label=label, linewidth=6, alpha=0.4, color=color)

                    if result["params"]["energy"] == result["params"]["energy"]:  # just to avoid duplicate entries
                        try:
                            # Reference curve
                            x_ref = df_ref["x"].values
                            y_ref = df_ref["y"].values
                    
                            # Model curve from pyMKM
                            x_model = result["data"]["dose"]
                            y_model = result["data"]["survival_fraction"]
                    
                            ref_idx = np.argsort(x_ref)
                            model_idx = np.argsort(x_model)
                            x_ref = x_ref[ref_idx]
                            y_ref = y_ref[ref_idx]
                            x_model = x_model[model_idx]
                            y_model = y_model[model_idx]
                    
                            metrics = log_linear_error_metrics(
                                x_ref=np.array(x_ref),
                                y_ref=np.array(y_ref),
                                x_model=np.array(x_model),
                                y_model=np.array(y_model)
                                )
                        
                            print(f"{cell_line}, Z = {Z}, LET = {let_val:.1f} | SMAPE_log = {metrics['smape_log']:.2f}%, "
                                  f"MeanLogError = {metrics['mean_log_error']:.3f}")
                    
                        except Exception as e:
                            print(f"{cell_line}, Z = {Z}, LET = {let_val:.1f} | Error during comparison: {type(e).__name__} - {e}")
                            metrics = {k: float('nan') for k in ['mean_log_error', 'rms_log_error', 'max_log_error', 'smape_log']}
                    
                        error_records.append({
                            'cell_line': cell_line,
                            'Z': Z,
                            'LET_MeV_cm': let_val,
                            'model': model_name,
                            'core_radius_type': core_type,
                            'domain_radius_um': domain_radius,
                            'nucleus_radius_um': nucleus_radius,
                            'MeanLogError': metrics['mean_log_error'],
                            'RMSLogError': metrics['rms_log_error'],
                            'MaxLogError': metrics['max_log_error'],
                            'SMAPE_log_percent': metrics['smape_log']
                        })

                    ax.set_xlim(0, 10)
                    ax.set_ylim(1e-4, 1)
                    ax.set_yscale("log")

                    x_label, y_label = map(str.strip, metadata.get("Data_Units", "Dose [Gy], Survival fraction").split(","))
                    ax.set_xlabel(x_label)
                    ax.set_ylabel(y_label)
                    ax.set_title(f"LET = {let_val:.1f} MeV/cm", fontsize=14)

                    if ax_idx == 0:
                        info_text = (
                            f"{cell_type}\n"
                            f"$\\alpha_0$: {alpha0:.3f} Gy$^{{-1}}$\n"
                            f"$\\beta_0$: {beta0:.3f} Gy$^{{-2}}$\n"
                            f"$r_d$: {domain_radius:.2f} μm\n"
                            f"$R_n$: {nucleus_radius:.2f} μm\n"
                            f"$z_0$: {z0_model:.2f} Gy"
                        )
                        ax.text(
                            0.05, 0.05, info_text, transform=ax.transAxes,
                            fontsize=14, verticalalignment='bottom', horizontalalignment='left',
                            bbox=dict(facecolor='white', alpha=0.85, edgecolor='black', boxstyle='round')
                        )

                    ax.grid(True, which='both', linestyle='--', alpha=0.3)
                    ax.legend(loc='upper right', fontsize=12)

                title = f"Source: {source}, Track model: {model_name} (Core: {core_type})"
                fig.suptitle(title, fontsize=16)
                fig.tight_layout(rect=[0, 0, 1, 0.95])
                fig_path = fig_dir / f"{cell_line}_Z{Z}_{source}.png"
                fig.savefig(fig_path, dpi=300)
                plt.pause(0.1)
                
    log_path = metrics_dir / f"sf_table_classic_metrics_{source}.csv"
    df_errors = pd.DataFrame(error_records)
    df_errors.to_csv(log_path, sep=csv_sep, index=False)
    print(f"\nSaved survival curve metrics to: {log_path}")

if __name__ == "__main__":
    validate_sf_table_classic(source="mstar_3_12")
    # validate_sf_table_classic(source="fluka_2020_0")
    # validate_sf_table_classic(source="geant4_11_3_0")
