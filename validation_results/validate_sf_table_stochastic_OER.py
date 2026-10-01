import matplotlib.pyplot as plt
from pathlib import Path
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import partial
# Access local pyMKM and validation utilities when executed directly.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pymkm.mktable.core import MKTableParameters, MKTable
from pymkm.sftable.core import SFTableParameters, SFTable
from pymkm.io.table_set import StoppingPowerTableSet
from pymkm.utils.parallel import optimal_worker_count
import warnings
from tqdm import tqdm
from validation_results.validation_utils.loader import load_validation_file
from validation_results.validation_utils.inverse_dose import inverse_dose_from_survival
from validation_results.validation_utils.metrics import semi_log_error_metrics

import locale
locale.setlocale(locale.LC_ALL, '')
csv_sep = ';' if locale.getlocale()[0] == 'Italian_Italy' else ','

import numpy as np
import pandas as pd

def _compute_d10_pair(sf_table, Z, energy, let):
    warnings.filterwarnings("ignore", category=UserWarning)
    sf_table.compute(ion=Z, energy=energy, let=let, force_recompute=True, apply_oxygen_effect=True)
    entry = sf_table.table[0]
    dose = entry["data"]["dose"]
    surv = entry["data"]["survival_fraction"]
    return energy, let, dose, surv

def validate_sf_table_stochastic_OER(source: str = "fluka_2020_0"):
    warnings.filterwarnings("ignore", category=UserWarning)
    base_dir = Path(__file__).resolve().parent / "sf_table_stochastic_OER"
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

            ref_results_cache = {}
            for pO2_dir in sorted(cell_line_dir.iterdir()):
                if not pO2_dir.is_dir():
                    continue

                grouped_by_Z = defaultdict(list)
                for txt_file in sorted(pO2_dir.glob("*.txt")):
                    metadata, df_ref = load_validation_file(txt_file)
                    Z = int(metadata["Atomic_Number"])
                    grouped_by_Z[Z].append((txt_file, metadata, df_ref))

                for Z in sorted(grouped_by_Z):
                    entries = grouped_by_Z[Z]
                    for txt_file, metadata, df_ref in entries:
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

                        sp_table_set = StoppingPowerTableSet.from_default_source(source).filter_by_ions([Z])
                        sp = sp_table_set.get(Z)
                        color = sp.color

                        mk_params = MKTableParameters(
                            domain_radius=domain_radius,
                            nucleus_radius=nucleus_radius,
                            z0=z0,
                            beta0=beta0,
                            model_name=model_name,
                            core_radius_type=core_type,
                            use_stochastic_model=True,
                        )
                        mk_table = MKTable(parameters=mk_params, sp_table_set=sp_table_set)

                        def build_sf_table(pO2_value):
                            sf_kwargs = dict(
                                mktable=mk_table,
                                beta0=beta0,
                                alphaS=alphaS,
                                alphaL=alphaL,
                                pO2=pO2_value,
                                K=K
                            )
                            if osmk_version == "2021":
                                sf_kwargs.update(
                                    zR=float(metadata["zR_Gy"]),
                                    gamma=float(metadata["gamma"]),
                                    Rm=float(metadata["Rm"]),
                                )
                            elif osmk_version == "2023":
                                sf_kwargs.update(
                                    f_rd_max=float(metadata["f_rd_max"]),
                                    f_z0_max=float(metadata["f_z0_max"]),
                                    Rmax=float(metadata["R_max"]),
                                )
                            return SFTable(parameters=SFTableParameters(**sf_kwargs))

                        sf_hyp = build_sf_table(pO2)
                        energies = sp.energy
                        lets = sp.let
                        pairs = list(zip(energies, lets))
                        worker_count = optimal_worker_count(pairs)

                        with ProcessPoolExecutor(max_workers=worker_count) as executor:
                            hyp_results = list(tqdm(
                                executor.map(partial(_compute_d10_pair, sf_hyp, Z), *zip(*pairs)),
                                total=len(pairs),
                                desc=f"Hypoxia Z={Z}, pO2={pO2:.1f}, Cell={metadata.get('Cell_Type', 'NA')}",
                                unit="energy"
                            ))

                        LET_vals = [let for _, let, _, _ in hyp_results]

                        if Z not in ref_results_cache:
                            sf_ref = build_sf_table(160.0)
                            with ProcessPoolExecutor(max_workers=worker_count) as executor:
                                ref_results = list(tqdm(
                                    executor.map(partial(_compute_d10_pair, sf_ref, Z), *zip(*pairs)),
                                    total=len(pairs),
                                    desc=f"Normoxia Z={Z}, Cell={metadata.get('Cell_Type', 'NA')}",
                                    unit="energy"
                                ))
                            ref_results_cache[Z] = ref_results
                        else:
                            ref_results = ref_results_cache[Z]

                        OER_vals = []
                        for (_, let, dose_hyp, surv_hyp), (_, _, dose_ref, surv_ref) in zip(hyp_results, ref_results):
                            d10_hyp = inverse_dose_from_survival(dose_hyp, surv_hyp, S_target=0.1)
                            d10_ref = inverse_dose_from_survival(dose_ref, surv_ref, S_target=0.1)
                            OER_vals.append(d10_hyp / d10_ref)

                        plt.figure(figsize=(7, 5))
                        ax = plt.gca()

                        # LET_SCALING = (4.0026032545 / 3.016029322) if Z == 2 else 1.0
                        LET_SCALING = 1.30 if metadata['Cell_Type'] == 'V79' else 1.0
                        # Add watermark if LET_SCALING != 1.0
                        if LET_SCALING != 1.0:
                           ax.text(
                               0.5, 0.5, f"LET scaled by {LET_SCALING:.2f}",
                               transform=ax.transAxes,
                               fontsize=30, color='gray', alpha=0.2,
                               ha='center', va='center', rotation=30
                           )

                        plt.plot(df_ref["x"]*LET_SCALING, df_ref["y"], '--', linewidth=2, color=color, label=metadata.get("Reference", "Reference"))

                        ion_symbol = sp.ion_symbol
                        model_suffix = f"- OSMK-{osmk_version}"
                        label = (f"{ion_symbol} (pyMKM {model_suffix})")
                        plt.plot(LET_vals, OER_vals, '-', linewidth=6, color=color, alpha=0.4, label=label)
                        
                        try:
                            x_model = np.array(LET_vals)
                            y_model = np.array(OER_vals)
                            x_ref = df_ref["x"].to_numpy() * LET_SCALING
                            y_ref = df_ref["y"].to_numpy()
                        
                            x_model, y_model = zip(*sorted(zip(x_model, y_model)))
                            x_ref, y_ref = zip(*sorted(zip(x_ref, y_ref)))
                        
                            metrics = semi_log_error_metrics(
                                x_ref=np.array(x_ref),
                                y_ref=np.array(y_ref),
                                x_model=np.array(x_model),
                                y_model=np.array(y_model)
                            )
                        
                            print(f"OER Z={Z}, {metadata['Cell_Type']}, pO₂={pO2:.1f} | SMAPE = {metrics['smape_percent']:.2f}%")
                        
                        except Exception as e:
                            print(f"OER Z={Z}, {metadata['Cell_Type']}, pO₂={pO2:.1f} | Error: {type(e).__name__} - {e}")
                            metrics = {k: float('nan') for k in ['mean_abs_error', 'rms_error', 'max_abs_error', 'smape_percent']}

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
                            'MeanAbsError': metrics['mean_abs_error'],
                            'RMS_Error': metrics['rms_error'],
                            'MaxAbsError': metrics['max_abs_error'],
                            'SMAPE_percent': metrics['smape_percent']
                        })

                        plt.xscale("log")
                        plt.yscale("linear")
                        plt.xlim(1E2, 1E4)
                        plt.ylim(1.0, max(OER_vals) * 1.1)
                        plt.xlabel("dose-averaged LET [MeV/cm]")
                        plt.ylabel("OER (10% Survival)")

                        cell_type = metadata.get("Cell_Type", f"Z{Z}")
                        title = f"Source: {source}, Track model: {model_name} (Core: {core_type})\nCell: {cell_type}"
                        plt.title(title,fontsize=10)

                        info_text = (
                            f"$pO_2$ = {pO2:.1f} mmHg\n"
                            f"$\\alpha_S$: {alphaS:.3f} | $\\alpha_L$: {alphaL:.3f}\n"
                            f"$\\beta_0$: {beta0:.4f} Gy$^{{-2}}$\n"
                            f"$r_d$: {domain_radius:.2f} μm | $R_n$: {nucleus_radius:.2f} μm\n"
                            f"$z_0$: {z0:.2f} Gy | K: {K:.1f} mmHg"
                        )
                        if osmk_version == "2021":
                            info_text += (
                                f"\n$z_R$: {float(metadata['zR_Gy']):.1f} Gy | $\\gamma$: {float(metadata['gamma']):.2f} | $R_m$: {float(metadata['Rm']):.2f}"
                            )
                        elif osmk_version == "2023":
                            info_text += (
                                f"\n$f^{{max}}_{{rd}}$: {float(metadata['f_rd_max'])} | $f^{{max}}_{{z_0}}$: {float(metadata['f_z0_max']):.2f} | $R_{{max}}$: {float(metadata['R_max']):.2f}"
                            )

                        plt.text(
                            0.05, 0.05, info_text, transform=plt.gca().transAxes,
                            fontsize=11, verticalalignment='bottom', horizontalalignment='left',
                            bbox=dict(facecolor='white', alpha=0.85, edgecolor='black', boxstyle='round')
                        )

                        plt.legend()
                        plt.tight_layout()
                        fig_path = fig_dir / f"OER_curve_{cell_type}_Z{Z}_pO2_{int(pO2)}_{source}_v{osmk_version}.png"
                        plt.savefig(fig_path, dpi=300)
                        plt.pause(0.1)
                        
    log_path = metrics_dir / f"sf_table_stochastic_OER_metrics_{source}.csv"
    df_errors = pd.DataFrame(error_records)
    df_errors.to_csv(log_path, sep=csv_sep, index=False)
    print(f"\nSaved OER metrics to: {log_path}")


if __name__ == "__main__":
    validate_sf_table_stochastic_OER(source="mstar_3_12")
    # validate_sf_table_stochastic_OER(source="fluka_2020_0")
    # validate_sf_table_stochastic_OER(source="geant4_11_3_0")
    
    
    
    
    