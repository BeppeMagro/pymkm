"""
Plotting utilities for survival-fraction (SF) curves.

This module defines :meth:`SFTable.plot`, which visualizes survival curves
computed with MKM, SMK, OSMK, or MCF-MKM for one or more ions.

Curves are displayed as semilogarithmic survival versus dose plots. The method
supports LET filtering and optional annotation of the active model parameters.
"""

import matplotlib.pyplot as plt
plt.rcParams.update({
    "axes.linewidth": 1.2,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "xtick.color": "black",
    "ytick.color": "black",
    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,
    "xtick.major.size": 5,
    "ytick.major.size": 5,
    "legend.fontsize": 14
})
from typing import Optional
from .core import SFTable

def plot(self,
         *,
         verbose: Optional[bool] = False,
         let: Optional[float] = None,
         ax: Optional[plt.Axes] = None,
         show: Optional[bool] = True
):
    """
    Plot survival fraction curves stored in ``self.table``.

    :param verbose: If True, displays model parameters on the plot.
    :type verbose: Optional[bool]
    :param let: LET value [MeV/cm] to filter the curves to plot. If None, all results are shown.
    :type let: Optional[float]
    :param ax: Matplotlib Axes object to draw on. If None, a new figure is created.
    :type ax: Optional[matplotlib.axes.Axes]
    :param show: If True, displays the plot. Set False when embedding or scripting.
    :type show: Optional[bool]

    :raises ValueError: If no results are available or no match is found for the specified LET.
    """
    
    if not self.table:
        raise ValueError("No survival data available. Run 'compute()' first.")

    results = self.table
    tolerance = 1e-3
    if let is not None:
        results_to_plot = [
            r for r in results if abs(r.get("params", {}).get("let", -999) - let) < tolerance
        ]
        if not results_to_plot:
            raise ValueError(f"No results found for LET = {let} MeV/cm.")
    else:
        results_to_plot = results

    for idx, result in enumerate(results_to_plot):
        params = result.get("params", {})
        calc_info = result.get("calculation_info", "N/A")
        df = result.get("data")

        if df is None or df.empty:
            print(f"\u26a0\ufe0f No data in result {idx + 1}.")
            continue

        ion = params.get("ion", "N/A")
        energy = params.get("energy", "N/A")
        let_val = params.get("let", "N/A")
        model = params.get("model", "N/A")

        color = self.params.mktable.sp_table_set.get(ion).color

        # Create figure/axes if not provided
        created_fig = False
        if ax is None:
            _, ax = plt.subplots()
            created_fig = True

        ax.plot(df["dose"], df["survival_fraction"],
                 label=f"{ion} | E={energy} MeV/u",
                 color=color, alpha=0.5, linewidth=6)

        ax.set_xlabel("Dose [Gy]", fontsize=14)
        ax.set_ylabel("Survival fraction", fontsize=14)
        ax.set_title(f"Survival Curve\nLET = {let_val} MeV/cm | Model: {model}", fontsize=16)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_xlim(left=0)
        ax.set_ylim(top=1)
        ax.set_yscale("log")
        ax.legend()

        if verbose and idx == 0:
            alpha0 = self.params.alpha0
            beta0 = self.params.beta0
            osmk_info = ""
            osmk_version = params.get("osmk_version")
            if osmk_version:
                pO2 = self.params.pO2
                osmk_info = f"\nOSMK: {osmk_version}, pO₂: {pO2:.2f} mmHg"
        
            info_text = (
                f"Model: {model}\n"
                f"α₀: {alpha0:.3f} Gy⁻¹\n"
                f"β₀: {beta0:.3f} Gy⁻²\n"
                f"Calculation: {calc_info}"
                f"{osmk_info}"
            )

            ax.text(0.05, 0.05, info_text, transform=ax.transAxes,
                    fontsize=12, verticalalignment='bottom', horizontalalignment='left',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='black', boxstyle='round'))

        if show and created_fig:
            plt.tight_layout()
            plt.show()

SFTable.plot = plot
