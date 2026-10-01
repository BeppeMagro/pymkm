import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import sys

# Extend path to allow local imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from validation_results.validation_utils.loader import load_validation_file
from validation_results.validation_utils.inverse_dose import compute_lq_dose_from_survival
from validation_results.validation_utils.metrics import semi_log_error_metrics
from pymkm.mktable.core import MKTableParameters, MKTable
from pymkm.io.table_set import StoppingPowerTableSet

import pandas as pd
import locale
locale.setlocale(locale.LC_ALL, '')
csv_sep = ';' if locale.getlocale()[0] == 'Italian_Italy' else ','

def validate_mk_table_classic_dose(source: str = "fluka_2020_0"):
    """
    Validate D10 values computed from MKTable using MKM-derived alpha and beta,
    and compare against reference data. For each ion, only LET values starting
    from the maximum LET are included.
    """
    # Setup paths
    base_dir = Path(__file__).resolve().parent / "mk_table_classic_dose"
    data_dir = base_dir / "reference_data"
    figure_dir = base_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = base_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Load and sort reference files by atomic number
    files = sorted(data_dir.glob("*.txt"), key=lambda f: int(load_validation_file(f)[0]["Atomic_Number"]))
    z_values = []
    file_map = {}
    error_records = []

    for file in files:
        metadata, _ = load_validation_file(file)
        Z = int(metadata["Atomic_Number"])
        z_values.append(Z)
        file_map[Z] = file

    # Load shared model parameters from the first reference file
    metadata, _ = load_validation_file(file_map[z_values[0]])
    rd = float(metadata["Domain_Radius_um"])
    rn = float(metadata["Nucleus_Radius_um"])
    beta0 = float(metadata["Beta0_Gy-2"])
    alpha0 = float(metadata["Alpha0_Gy-1"])
    model_name = metadata["Model_Name"]
    core_type = metadata["Core_Radius_Type"]
    S = float(metadata["Survival_Fraction"])

    # Initialize MKTable and compute for selected ions
    sp_table_set = StoppingPowerTableSet.from_default_source(source).filter_by_ions(z_values)
    params = MKTableParameters(
        domain_radius=rd,
        nucleus_radius=rn,
        beta0=beta0,
        model_name=model_name,
        core_radius_type=core_type
    )
    mk_table = MKTable(parameters=params, sp_table_set=sp_table_set)
    mk_table.compute(ions=z_values)

    # Plotting
    fig, ax = plt.subplots(figsize=(8, 6))

    for Z in z_values:
        metadata, df_ref = load_validation_file(file_map[Z])
        table = mk_table.get_table(Z)

        let = table["let"].to_numpy()
        z_bar_star_domain = table["z_bar_star_domain"].to_numpy()

        # Get reference LET range
        x_ref = df_ref["x"].to_numpy()
        let_min_ref = np.min(x_ref) * 0.95
        let_max_ref = np.max(x_ref) * 1.05

        # Start from index of maximum LET in pyMKM
        idx_max_let = np.argmax(let)

        # Slice from max LET onwards
        let = let[idx_max_let:]
        z_bar_star_domain = z_bar_star_domain[idx_max_let:]

        # Then apply LET bounds from reference data
        mask = (let >= let_min_ref) & (let <= let_max_ref)
        let = let[mask]
        z_bar_star_domain = z_bar_star_domain[mask]

        # Compute D10 using MKM model
        alpha_mkm = alpha0 + beta0 * z_bar_star_domain
        beta_mkm = np.full_like(alpha_mkm, beta0)
        d10 = np.array([
            compute_lq_dose_from_survival(a, b, S) for a, b in zip(alpha_mkm, beta_mkm)
        ])
        
        # Compute metrics comparing D10 vs LET (semi-log domain)
        try:
            y_model = d10
            x_model = let
            x_ref = df_ref["x"].to_numpy()
            y_ref = df_ref["y"].to_numpy()
            
            # Sort both datasets by increasing LET (x)
            sorted_model_idx = np.argsort(x_model)
            x_model = x_model[sorted_model_idx]
            y_model = y_model[sorted_model_idx]
            
            sorted_ref_idx = np.argsort(x_ref)
            x_ref = x_ref[sorted_ref_idx]
            y_ref = y_ref[sorted_ref_idx]
        
            metrics = semi_log_error_metrics(x_ref, y_ref, x_model, y_model)
            print(f"Z = {Z} | mean = {metrics['mean_abs_error']:.3f}, "
                  f"rms = {metrics['rms_error']:.3f}, "
                  f"max = {metrics['max_abs_error']:.3f}, "
                  f"SMAPE = {metrics['smape_percent']:.2f}%")
        except Exception as e:
            print(f"Z = {Z} | Error during comparison: {e}")
            metrics = {k: float('nan') for k in ['mean_abs_error', 'rms_error', 'max_abs_error', 'smape_percent']}

        error_records.append({
            'atomic_number': Z,
            'model': model_name,
            'core_radius_type': core_type,
            'domain_radius_um': rd,
            'nucleus_radius_um': rn,
            'MeanAbsError': metrics['mean_abs_error'],
            'RMS_Error': metrics['rms_error'],
            'MaxAbsError': metrics['max_abs_error'],
            'SMAPE_percent': metrics['smape_percent']
        })

        # Plot pyMKM result (solid, thick, semi-transparent)
        ion_symbol = sp_table_set.get(Z).ion_symbol
        color = sp_table_set.get(Z).color
        ax.plot(let, d10, label=f"{ion_symbol} (pyMKM)", color=color, alpha=0.4, linewidth=6)

    for Z in z_values:
            # Plot reference data (dashed)
            metadata, df_ref = load_validation_file(file_map[Z])
            x_ref = df_ref["x"].to_numpy()
            y_ref = df_ref["y"].to_numpy()
            color = sp_table_set.get(Z).color
            ref_label = metadata.get("Reference", "Unknown")
            ax.plot(x_ref, y_ref, '--', label=ref_label, color=color, linewidth=3)
        
    # Title and labels
    title = f"Source: {source}, Track model: {model_name} (Core: {core_type})"
    ax.set_title(title)
    ax.set_xscale("log")
    ax.set_xlim(1e2, 1e4)
    ax.set_ylim(0, 3.5)
    ax.set_xlabel("LET [MeV/cm]")
    ax.set_ylabel(r"$D_{10}$ [Gy]")

    # Info box with parameters
    z0 = mk_table.params.z0
    info_text = (
        f"$\\alpha_0$: {alpha0:.3f} Gy$^{{-1}}$\n"
        f"$\\beta_0$: {beta0:.3f} Gy$^{{-2}}$\n"
        f"$r_d$: {rd:.2f} μm\n"
        f"$R_n$: {rn:.2f} μm\n"
        f"$z_0$: {z0:.2f} Gy"
    )
    ax.text(
        0.05, 0.05, info_text, transform=ax.transAxes,
        fontsize=14, verticalalignment='bottom', horizontalalignment='left',
        bbox=dict(facecolor='white', alpha=0.85, edgecolor='black', boxstyle='round')
    )

    # Legend and layout
    ax.legend(ncol=2, loc='lower right', frameon=True, fontsize=12)
    fig.tight_layout()

    # Save and show
    out_path = figure_dir / f"D10_vs_LET_{source}.png"
    fig.savefig(out_path, dpi=300)
    plt.show(block=False)
    plt.pause(0.1)
    
    log_path = metrics_dir / f"mk_table_classic_dose_metrics_{source}.csv"
    df_errors = pd.DataFrame(error_records)
    df_errors.to_csv(log_path, sep=csv_sep, index=False)
    print(f"\nSaved validation metrics to: {log_path}")

if __name__ == "__main__":
    validate_mk_table_classic_dose(source="mstar_3_12")
    # validate_mk_table_classic_dose(source="fluka_2020_0")
    # validate_mk_table_classic_dose(source="geant4_11_3_0")

