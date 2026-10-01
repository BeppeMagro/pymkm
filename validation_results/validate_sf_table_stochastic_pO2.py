import matplotlib.pyplot as plt
from pathlib import Path
import sys
import numpy as np
from collections import defaultdict

# Extend module search path to allow relative imports
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

def validate_sf_table_stochastic_pO2(source: str = "fluka_2020_0"):
    """
    Validate survival fraction curves computed by SFTable (SMK mode with oxygen effect)
    against reference datasets under hypoxic/normoxic conditions.
    """
    base_dir = Path(__file__).resolve().parent / "sf_table_stochastic_pO2"
    ref_dir = base_dir / "reference_data"
    fig_dir = base_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = base_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    error_records = []
    for osmk_version in ["2021", "2023"]:
        version_dir = ref_dir / osmk_version
        if not version_dir.exists():
            continue

        for cell_line_dir in sorted(version_dir.iterdir()):
            if not cell_line_dir.is_dir():
                continue

            for pO2_dir in sorted(cell_line_dir.iterdir()):
                if not pO2_dir.is_dir():
                    continue

                grouped_by_Z = defaultdict(list)
                for txt_file in sorted(pO2_dir.glob("*.txt")):
                    metadata, df_ref = load_validation_file(txt_file)
                    Z = int(metadata["Atomic_Number"])
                    let_val = float(metadata["LET_MeV_cm"])
                    grouped_by_Z[Z].append((let_val, txt_file, metadata, df_ref))

                for Z in sorted(grouped_by_Z):
                    entries = grouped_by_Z[Z]
                    entries.sort(key=lambda x: x[0])
                    layout_list = choose_horizontal_subplot_layout(len(entries), max_cols_per_fig=3)
                    entry_idx = 0

                    for fig_idx, (n_rows, n_cols) in enumerate(layout_list):
                        fig, axs = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5), squeeze=False)
                        axs = axs[0]

                        for ax_idx, ax in enumerate(axs):
                            if entry_idx >= len(entries):
                                ax.axis("off")
                                continue

                            let_val, txt_file, metadata, df_ref = entries[entry_idx]
                            entry_idx += 1

                            domain_radius = float(metadata["Domain_Radius_um"])
                            nucleus_radius = float(metadata["Nucleus_Radius_um"])
                            z0 = float(metadata["z0_Gy"])
                            beta0 = float(metadata["Beta0_Gy-2"])
                            alphaS = float(metadata.get("AlphaS_Gy-1"))
                            alphaL = float(metadata.get("AlphaL_Gy-1"))
                            K = float(metadata["K_mmHg"])
                            pO2 = float(metadata["pO2_mmHg"])
                            model_name = metadata["Model_Name"]
                            core_type = metadata["Core_Radius_Type"]
                            atomic_number = int(metadata["Atomic_Number"])
                            
                            # LET_SCALING = (4.0026032545 / 3.016029322) if atomic_number == 2 else 1.0
                            LET_SCALING = 1.3 if metadata['Cell_Type'] == 'V79' else 1.0
                            # Add watermark if LET_SCALING != 1.0
                            if LET_SCALING != 1.0:
                               ax.text(
                                   0.5, 0.5, f"LET scaled by {LET_SCALING:.2f}",
                                   transform=ax.transAxes,
                                   fontsize=30, color='gray', alpha=0.2,
                                   ha='center', va='center', rotation=30
                               )


                            sp_table_set = StoppingPowerTableSet.from_default_source(source).filter_by_ions([Z])

                            mk_params = MKTableParameters(
                                domain_radius=domain_radius,
                                nucleus_radius=nucleus_radius,
                                z0=z0,
                                beta0=beta0,
                                model_name=model_name,
                                core_radius_type=core_type,
                                use_stochastic_model=True
                            )
                            mk_table = MKTable(parameters=mk_params, sp_table_set=sp_table_set)

                            dose_grid = np.asarray(df_ref["x"].values, dtype=float)

                            sf_kwargs = dict(
                                mktable=mk_table,
                                beta0=beta0,
                                dose_grid=dose_grid,
                                alphaS=alphaS,
                                alphaL=alphaL,
                                pO2=pO2,
                                K=K
                            )

                            if osmk_version == "2021":
                                sf_kwargs.update(dict(
                                    zR=float(metadata["zR_Gy"]),
                                    gamma=float(metadata["gamma"]),
                                    Rm=float(metadata["Rm"]),
                                ))
                            elif osmk_version == "2023":
                                sf_kwargs.update(dict(
                                    f_rd_max=float(metadata["f_rd_max"]),
                                    f_z0_max=float(metadata["f_z0_max"]),
                                    Rmax=float(metadata["R_max"]),
                                ))

                            sf_table = SFTable(parameters=SFTableParameters(**sf_kwargs))
                            # Correct 4He let for mass(4He)/mass(3He)
                            let_val *= LET_SCALING
                            #
                            sf_table.compute(ion=atomic_number, let=let_val, force_recompute=True, apply_oxygen_effect=True)

                            color = sp_table_set.get(atomic_number).color
                            ax.plot(dose_grid, df_ref["y"].values, '--', label=metadata.get("Reference", "Reference"), linewidth=3, color=color)

                            for idx, result in enumerate(sf_table.table):
                                ion_symbol = sp_table_set.get(atomic_number).ion_symbol
                                model_suffix = f"- OSMK-{osmk_version}"
                                label = (f"{ion_symbol} (pyMKM {model_suffix})")
                                line_style = '-' if idx == 0 else ':'
                                ax.plot(result["data"]["dose"], result["data"]["survival_fraction"], line_style, label=label, linewidth=6, alpha=0.4, color=color)

                                if idx == 0:
                                    try:
                                        x_ref = np.asarray(df_ref["x"].values, dtype=float)
                                        y_ref = np.asarray(df_ref["y"].values, dtype=float)
                                        x_model = np.asarray(result["data"]["dose"])
                                        y_model = np.asarray(result["data"]["survival_fraction"])
                                
                                        x_ref, y_ref = zip(*sorted(zip(x_ref, y_ref)))
                                        x_model, y_model = zip(*sorted(zip(x_model, y_model)))
                                
                                        metrics = log_linear_error_metrics(
                                            x_ref=np.array(x_ref),
                                            y_ref=np.array(y_ref),
                                            x_model=np.array(x_model),
                                            y_model=np.array(y_model)
                                        )
                                
                                        print(f"O2-SF: Cell = {metadata['Cell_Type']}, Z = {Z}, pO₂ = {pO2:.1f} | SMAPE = {metrics['smape_log']:.2f} %")
                                
                                    except Exception as e:
                                        print(f"Error SF metric @ Cell = {metadata['Cell_Type']}, Z = {Z}, pO₂ = {pO2:.1f}: {type(e).__name__} - {e}")
                                        metrics = {k: float('nan') for k in ['mean_log_error', 'rms_log_error', 'max_log_error', 'smape_log']}
                                
                                    error_records.append({
                                        'cell_type': metadata['Cell_Type'],
                                        'Z': Z,
                                        'pO2_mmHg': pO2,
                                        'model': model_name,
                                        'core_radius_type': core_type,
                                        'domain_radius_um': domain_radius,
                                        'nucleus_radius_um': nucleus_radius,
                                        'osmk_version': osmk_version,
                                        'LET_scaling': LET_SCALING,
                                        'LET_MeV_cm': let_val,
                                        'MeanLogError': metrics['mean_log_error'],
                                        'RMSLogError': metrics['rms_log_error'],
                                        'MaxLogError': metrics['max_log_error'],
                                        'SMAPE_log_percent': metrics['smape_log']
                                    })

                            ax.set_xlim(0, 10)
                            ax.set_ylim(1e-4, 1)
                            ax.set_yscale("log")
                            ax.set_xlabel("Dose [Gy]")
                            ax.set_ylabel("Survival fraction")
                            ax.set_title(f"LET = {let_val:.1f} MeV/cm, E = {result['params']['energy']:.1f} MeV/u",
                                         fontsize=14)

                            info_text = (
                                f"$pO_2$ = {pO2:.1f} mmHg\n"
                                f"$\\alpha_S$: {alphaS:.3f} | $\\alpha_L$: {alphaL:.3f}\n"
                                f"$\\beta_0$: {beta0:.4f} Gy$^{{-2}}$\n"
                                f"$r_d$: {domain_radius:.2f} μm | $R_n$: {nucleus_radius:.2f} μm\n"
                                f"$z_0$: {z0:.2f} Gy | K: {K:.1f} mmHg"
                            )

                            if osmk_version == "2021":
                                info_text += (
                                    f"\n$z_R$: {sf_kwargs['zR']:.1f} Gy | $\\gamma$: {sf_kwargs['gamma']:.2f} | $R_m$: {sf_kwargs['Rm']:.2f}"
                                )
                            elif osmk_version == "2023":
                                info_text += (
                                    f"\n$f^{{max}}_{{rd}}$: {sf_kwargs['f_rd_max']:.2f} | $f^{{max}}_{{z_0}}$: {sf_kwargs['f_z0_max']:.2f} | $R_{{max}}$: {sf_kwargs['Rmax']:.2f}"
                                )

                            if ax_idx == 0:
                                ax.text(
                                    0.05, 0.05, info_text, transform=ax.transAxes,
                                    fontsize=13, verticalalignment='bottom', horizontalalignment='left',
                                    bbox=dict(facecolor='white', alpha=0.85, edgecolor='black', boxstyle='round')
                                )

                            ax.grid(True, which='both', linestyle='--', alpha=0.3)
                            ax.legend(loc='upper right', fontsize=11)

                        title = f"Source: {source}, Track model: {model_name} (Core: {core_type})\nCell: {metadata['Cell_Type']}"
                        fig.suptitle(title, fontsize=10 if (n_rows * n_cols) == 1 else 14)
                        fig.tight_layout(rect=[0, 0, 1, 0.95])

                        fig_name = f"{metadata['Cell_Type']}_Z{Z}_pO2_{int(pO2)}_{source}_v{osmk_version}_fig{fig_idx}.png"
                        fig_path = fig_dir / fig_name
                        fig.savefig(fig_path, dpi=300)
                        plt.pause(0.1)
                        
    log_path = metrics_dir / f"sf_table_stochastic_pO2_metrics_{source}.csv"
    df_errors = pd.DataFrame(error_records)
    df_errors.to_csv(log_path, sep=csv_sep, index=False)
    print(f"\nSaved O₂ survival metrics to: {log_path}")


if __name__ == "__main__":
    validate_sf_table_stochastic_pO2(source="mstar_3_12")
    # validate_sf_table_stochastic_pO2(source="fluka_2020_0")
    # validate_sf_table_stochastic_pO2(source="geant4_11_3_0")


