#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Appendix B1 – Comparison of z̄_d* (dose-mean specific energy) across MSTAR, FLUKA, and Geant4.

This script:
 - Loads stopping power tables for ions with Z ≤ 6 (H–C) from MSTAR, FLUKA, and Geant4.
 - Resamples FLUKA and Geant4 onto the MSTAR energy grid (common overlap).
 - Computes z̄_d* for each ion/source using MKTable.
 - Calculates SMAPE (%) between each source and MSTAR for every ion and energy point.
 - Reports average, minimum, and maximum SMAPE over all ions and energies.
 - Plots z̄_d* vs Energy (log-x scale) for all sources.

Requires pyMKM ≥ 0.1.0
"""

import numpy as np
import matplotlib.pyplot as plt

from pymkm.io.table_set import StoppingPowerTableSet
from pymkm.mktable.core import MKTable, MKTableParameters

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
IONS = [1, 2, 3, 4, 5, 6]  # H–C
SOURCES = ["mstar_3_12", "fluka_2020_0", "geant4_11_3_0"]

COLORS = {"mstar_3_12": "#1f77b4", "fluka_2020_0": "#ff7f0e", "geant4_11_3_0": "#2ca02c"}
STYLES = {"mstar_3_12": "-", "fluka_2020_0": "--", "geant4_11_3_0": ":"}
FIG_OUT = "appendix_B1_zeta_only.png"

# MKM parameters (same as in your examples)
MK_PARAMS = MKTableParameters(
    domain_radius=0.32,
    nucleus_radius=3.9,
    beta0=0.0615,
    model_name="Kiefer-Chatterjee",
    core_radius_type="energy-dependent",
    use_stochastic_model=False,
)

# ------------------------------------------------------------
# Metric: Symmetric Mean Absolute Percentage Error (SMAPE)
# ------------------------------------------------------------
def smape(y_pred: np.ndarray, y_true: np.ndarray, eps: float = 1e-12) -> float:
    """
    Symmetric Mean Absolute Percentage Error (in %):
        SMAPE = (100/N) * sum( |y' - y| / (|y| + |y'|) )
    Uses linear Y values and log-scale X (energy axis).
    """
    num = np.abs(y_pred - y_true)
    den = np.maximum(np.abs(y_true) + np.abs(y_pred), eps)
    return float(np.mean(num / den) * 100.0)

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def load_table_set(source: str, Z: int) -> StoppingPowerTableSet:
    """Load and return a filtered TableSet for a given source and ion."""
    ts = StoppingPowerTableSet.from_default_source(source)
    return ts.filter_by_ions([str(Z)])

def common_energy_grid(mstar, fluka, geant):
    """Compute common energy grid from overlap; take MSTAR points within [lo, hi]."""
    lo = max(mstar.energy_grid.min(), fluka.energy_grid.min(), geant.energy_grid.min())
    hi = min(mstar.energy_grid.max(), fluka.energy_grid.max(), geant.energy_grid.max())
    grid = mstar.energy_grid[(mstar.energy_grid >= lo) & (mstar.energy_grid <= hi)]
    grid = np.unique(grid)
    if grid.size < 2:
        raise RuntimeError("Common energy grid too small – check overlap among datasets.")
    return grid

def compute_zeta_d_star_from_table(tbl):
    """Compute z̄_d* from a single-ion StoppingPowerTable."""
    single_set = StoppingPowerTableSet()
    single_set.add(str(tbl.atomic_number), tbl)
    mk = MKTable(parameters=MK_PARAMS, sp_table_set=single_set)
    mk.compute(ions=single_set.get_available_ions(), parallel=False)
    ion_name = single_set.get_available_ions()[0]
    df = mk.get_table(ion_name)
    return df["energy"].to_numpy(), df["z_bar_star_domain"].to_numpy()

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    fig, ax = plt.subplots(figsize=(6.0, 4.2), dpi=180)

    smape_results = {"fluka_2020_0": [], "geant4_11_3_0": []}
    per_ion_stats = {}

    for Z in IONS:
        # Load table sets for current ion
        mstar_set = load_table_set("mstar_3_12", Z)
        fluka_set = load_table_set("fluka_2020_0", Z)
        geant_set = load_table_set("geant4_11_3_0", Z)

        mstar = mstar_set.get(str(Z))
        fluka = fluka_set.get(str(Z))
        geant = geant_set.get(str(Z))

        # Common grid and resampling
        Ecom = common_energy_grid(mstar, fluka, geant)
        mstar.resample(Ecom)
        fluka.resample(Ecom)
        geant.resample(Ecom)

        # Compute z̄_d* for each source
        e_ref, z_ref = compute_zeta_d_star_from_table(mstar)
        e_f, z_f = compute_zeta_d_star_from_table(fluka)
        e_g, z_g = compute_zeta_d_star_from_table(geant)

        # Plot curves
        ax.plot(e_ref, z_ref, ls=STYLES["mstar_3_12"], color=COLORS["mstar_3_12"],
                lw=1.5, alpha=0.9, label="MSTAR" if Z == 6 else None)
        ax.plot(e_f, z_f, ls=STYLES["fluka_2020_0"], color=COLORS["fluka_2020_0"],
                lw=1.2, alpha=0.9, label="FLUKA" if Z == 6 else None)
        ax.plot(e_g, z_g, ls=STYLES["geant4_11_3_0"], color=COLORS["geant4_11_3_0"],
                lw=1.2, alpha=0.9, label="Geant4" if Z == 6 else None)

        # SMAPE vs MSTAR
        sm_fluka = smape(z_f, z_ref)
        sm_geant = smape(z_g, z_ref)
        smape_results["fluka_2020_0"].append(sm_fluka)
        smape_results["geant4_11_3_0"].append(sm_geant)

        per_ion_stats[Z] = {
            "fluka": sm_fluka,
            "geant4": sm_geant
        }

    # ---- Plot styling
    ax.set_xscale("log")
    ax.set_xlabel("Energy [MeV/u]")
    ax.set_ylabel(r"$\bar{z}_d^*$")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(frameon=False, loc="lower left", fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_OUT, bbox_inches="tight", dpi=300)
    plt.show()

    # ---- Compute global statistics: mean, min, max SMAPE
    def stats(values):
        arr = np.array(values)
        return float(np.mean(arr)), float(np.min(arr)), float(np.max(arr))

    global_stats = {
        src: stats(vals) for src, vals in smape_results.items()
    }

    # ---- Print results
    print("\n==== SMAPE (%) for z̄_d* vs MSTAR ====")
    print("(Mean, Min, Max across all ions and energies)\n")
    for src, (mean_, min_, max_) in global_stats.items():
        print(f"{src:14s}: mean = {mean_:6.3f} %,  min = {min_:6.3f} %,  max = {max_:6.3f} %")

    print("\n---- Per-ion average SMAPE (%) ----")
    for Z, vals in per_ion_stats.items():
        print(f"Z={Z}:  FLUKA={vals['fluka']:.3f} %,  Geant4={vals['geant4']:.3f} %")

    print(f"\n✅ Figure saved to: {FIG_OUT}")

# ------------------------------------------------------------
if __name__ == "__main__":
    main()
