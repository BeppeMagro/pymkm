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

def validate_mk_table_classic(source: str = "fluka_2020_0"):
    """
    Validate the dose-mean saturation-corrected specific energy z̄* computed by MKTable
    against reference datasets from Inaniwa et al. (2010), for various ions and energies.
    """
    data_dir = Path(__file__).resolve().parent / "mk_table_classic" / "reference_data"
    figure_dir = Path(__file__).resolve().parent / "mk_table_classic" / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = Path(__file__).resolve().parent / "mk_table_classic" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(data_dir.glob("*.txt"))
 
    error_records = []
    for file in files:
        metadata, df_ref = load_validation_file(file)

        label = f"{metadata.get('Reference', 'Unknown')}"
        atomic_number = int(metadata["Atomic_Number"])
        model_name = metadata["Model_Name"]
        core_type = metadata["Core_Radius_Type"]
        domain_radius = float(metadata["Domain_Radius_um"])
        nucleus_radius = float(metadata["Nucleus_Radius_um"])
        beta0 = float(metadata["Beta0_Gy-2"])
        
        title = f"Source: {source}, Track model: {model_name} (Core: {core_type})"

        
        print(f"\nValidating ion Z = {atomic_number} using source '{source}'...")
        sp_table_set = StoppingPowerTableSet.from_default_source(source).filter_by_ions([atomic_number])

        params = MKTableParameters(
            domain_radius=domain_radius,
            nucleus_radius=nucleus_radius,
            beta0=beta0,
            model_name=model_name,
            core_radius_type=core_type,
        )

        mk_table = MKTable(parameters=params, sp_table_set=sp_table_set)
        mk_table.compute(ions=[atomic_number])

        # Plot model result using built-in method
        fig, ax = plt.subplots(figsize=(10, 6))
        mk_table.plot(
            ions=[atomic_number],
            x="energy",
            y="z_bar_star_domain",
            verbose=True,
            ax=ax,
            show=False
            )

        ax.plot(
            df_ref['x'], df_ref['y'],
            marker='o', linestyle='None',
            markersize=9, markeredgewidth=0.8,
            markeredgecolor='black', alpha=0.7,
            label=label,
            color=sp_table_set.get(atomic_number).color
        )
        
        ax.set_title(title, fontsize=14)
        ax.legend(fontsize=12)
        fig.tight_layout()
        
        plot_file = figure_dir / f"z_bar_star_domain_Z{atomic_number}_{source}.png"
        fig.savefig(plot_file, dpi=300)
        plt.close(fig)
        
        # Interpolate model vs reference data on log-energy domain
        try:
            x_ref = np.asarray(df_ref['x'], dtype=float)
            y_ref = np.asarray(df_ref['y'], dtype=float)
            
            df_model = mk_table.get_table(atomic_number)
            x_model = df_model["energy"].values
            y_model = df_model["z_bar_star_domain"].values
        
            metrics = semi_log_error_metrics(x_ref, y_ref, x_model, y_model)
            print(f"{file.name} | Errors: "
                  f"mean={metrics['mean_abs_error']:.3f}, "
                  f"rms={metrics['rms_error']:.3f}, "
                  f"max={metrics['max_abs_error']:.3f}, "
                  f"SMAPE={metrics['smape_percent']:.2f}%")
        except Exception as e:
            print(f"{file.name} | Error during comparison: {e}")
            metrics = {k: np.nan for k in ['mean_abs_error', 'rms_error', 'max_abs_error', 'smape_percent']}
        
        # Save for CSV output
        error_records.append({
            'filename': file.name,
            'atomic_number': atomic_number,
            'model': model_name,
            'core_radius_type': core_type,
            'domain_radius_um': domain_radius,
            'nucleus_radius_um': nucleus_radius,
            'MeanAbsError': metrics['mean_abs_error'],
            'RMS_Error': metrics['rms_error'],
            'MaxAbsError': metrics['max_abs_error'],
            'SMAPE_percent': metrics['smape_percent']
        })
               
    log_path = metrics_dir / f"mk_table_classic_metrics_{source}.csv"
    df_errors = pd.DataFrame(error_records)
    df_errors.to_csv(log_path, sep=csv_sep, index=False)
    print(f"\nSaved validation metrics to: {log_path}")
                
if __name__ == "__main__":
    validate_mk_table_classic(source="mstar_3_12")
    # validate_mk_table_classic(source="fluka_2020_0")
    # validate_mk_table_classic(source="geant4_11_3_0")
