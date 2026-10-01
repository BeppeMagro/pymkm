import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Access to local modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from validation_results.validation_utils.loader import load_validation_file
from validation_results.validation_utils.metrics import semi_log_error_metrics

from pymkm.mktable.core import MKTableParameters, MKTable
from pymkm.io.table_set import StoppingPowerTableSet

import pandas as pd
import numpy as np
import locale

locale.setlocale(locale.LC_ALL, '')
csv_sep = ';' if locale.getlocale()[0] == 'Italian_Italy' else ','

def validate_mk_table_stochastic(source: str = "fluka_2020_0"):
    """
    Validate (z̄_d, z̄*, z̄_n) by MKTable
    against reference datasets from Inaniwa et al. (2018), for various ions and energies.
    """
    base_dir = Path(__file__).resolve().parent / "mk_table_stochastic"
    ref_dir = base_dir / "reference_data"
    fig_dir = base_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = base_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)


    subfolders = {
        "domain": "mk_table_avg_domain",
        "saturation": "mk_table_avg_saturation_domain",
        "nucleus": "mk_table_avg_nucleus"
    }

    # Organize reference files by Z
    reference_data = {}
    for key, folder in subfolders.items():
        path = ref_dir / folder
        for txt_file in path.glob("*.txt"):
            Z = int(txt_file.name.split("_Z")[1].split("_")[0])
            if Z not in reference_data:
                reference_data[Z] = {}
            reference_data[Z][key] = txt_file

    error_records = []
    for Z, files in sorted(reference_data.items()):
        metadata, _ = load_validation_file(files["domain"])
        
        label = f"{metadata.get('Reference', 'Unknown')}"
        atomic_number = int(metadata["Atomic_Number"])
        model_name = metadata["Model_Name"]
        core_type = metadata["Core_Radius_Type"]
        domain_radius = float(metadata["Domain_Radius_um"])
        nucleus_radius = float(metadata["Nucleus_Radius_um"])
        z0 = float(metadata["z0_Gy"])
        model_version = metadata.get("Model_Version", "classic").lower()

        params = MKTableParameters(
            domain_radius=domain_radius,
            nucleus_radius=nucleus_radius,
            z0=z0,
            model_name=model_name,
            core_radius_type=core_type,
            use_stochastic_model=model_version == "stochastic"
            )

        print(f"\nValidating ion Z = {atomic_number} using source '{source}'...")
        sp_table_set = StoppingPowerTableSet.from_default_source(source).filter_by_ions([atomic_number])
        mk_table = MKTable(parameters=params, sp_table_set=sp_table_set)
        mk_table.compute(ions=[atomic_number])

        fig, axs = plt.subplots(1, 3, figsize=(18, 5))
        labels = {
            "domain": r"$\bar{z}_d$",
            "saturation": r"$\bar{z}^*$",
            "nucleus": r"$\bar{z}_n$"
        }
        keys = {
            "domain": "z_bar_domain",
            "saturation": "z_bar_star_domain",
            "nucleus": "z_bar_nucleus"
        }

        for ax, key in zip(axs, ["domain", "saturation", "nucleus"]):
            _, df_ref = load_validation_file(files[key])
            df_model = mk_table.get_table(Z)
            x_model = df_model["energy"]
            y_model = df_model[keys[key]]
            
            sorted_ref_idx = np.argsort(df_ref['x'].values)
            x_ref = df_ref['x'].values[sorted_ref_idx]
            y_ref = df_ref['y'].values[sorted_ref_idx]
            
            sorted_model_idx = np.argsort(x_model.values)
            x_model_sorted = x_model.values[sorted_model_idx]
            y_model_sorted = y_model.values[sorted_model_idx]
            
            try:
                metrics = semi_log_error_metrics(x_ref, y_ref, x_model_sorted, y_model_sorted)
                print(f"Z = {Z}, {key} | SMAPE = {metrics['smape_percent']:.2f} %")
            except Exception as e:
                print(f"Z = {Z}, {key} | Error: {type(e).__name__} - {e}")
                metrics = {k: float('nan') for k in ['mean_abs_error', 'rms_error', 'max_abs_error', 'smape_percent']}

            error_records.append({
                'Z': Z,
                'quantity': key,
                'model': model_name,
                'core_radius_type': core_type,
                'domain_radius_um': domain_radius,
                'nucleus_radius_um': nucleus_radius,
                'source': source,
                'MeanAbsError': metrics['mean_abs_error'],
                'RMS_Error': metrics['rms_error'],
                'MaxAbsError': metrics['max_abs_error'],
                'SMAPE_percent': metrics['smape_percent']
            })
            
            color = mk_table.sp_table_set.get(atomic_number).color
            ax.plot(x_model, y_model, label=f"pyMKM (Z = {atomic_number})", color=color, alpha=0.5, linewidth=6)
            ax.plot(df_ref['x'], df_ref['y'], marker='o', linestyle='None', 
                     markersize=9, markeredgewidth=0.8, markeredgecolor='black', alpha=0.7,
                     label=label, color=color)
            
            x_min = min(x_model)
            x_max = max(x_model)
            y_max = max(y_model)            
            ax.set_ylim(0, y_max * 1.05)
            ax.set_xlim(x_min, x_max)
            ax.set_xscale("log")
            ax.set_xlabel("Energy [MeV/u]")
            ax.set_ylabel(labels[key] + " [Gy]")
            ax.grid(True, which='both', linestyle='--', alpha=0.1)
            ax.legend()    

        fig.suptitle(f"Source: {source}, Track model: {model_name} (Core: {core_type})\n$r_d$: {domain_radius:.2f} $μm$, $R_n$: {nucleus_radius:.1f} $μm$, $z_0$: {z0:.2f} Gy", fontsize=14)

        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(fig_dir / f"z_bar_all_Z{Z}_{source}.png", dpi=300)
        
    log_path = metrics_dir / f"mk_table_stochastic_metrics_{source}.csv"
    df_errors = pd.DataFrame(error_records)
    df_errors.to_csv(log_path, sep=csv_sep, index=False)
    print(f"\nSaved validation metrics to: {log_path}")
        
if __name__ == "__main__":
    validate_mk_table_stochastic(source="mstar_3_12")
    # validate_mk_table_stochastic(source="fluka_2020_0")
    # validate_mk_table_stochastic(source="geant4_11_3_0")
