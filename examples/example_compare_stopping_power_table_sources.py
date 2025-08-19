#!/usr/bin/env python
"""
Example usage of StoppingPowerTableSet to compare stopping power tables from different default sources.

This script demonstrates how to:
  - Load stopping power tables from the different default sources ("fluka_2020_0") for all available ions.
  - Display the total number of loaded tables and the available ion keys.
  - Plot the stopping power curves to compare different sources.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from pymkm.io.data_registry import get_available_sources
from pymkm.io.table_set import StoppingPowerTableSet

def main():

    ## Load stopping power table
    sources = get_available_sources()
    print("Loading stopping power tables from default sources:...")

    _, ax = plt.subplots(figsize=(10, 6))
    linestyles = ['-', '--', '']
    markers = [None, None, 's']
    custom_lines = []

    for i, source in enumerate(sources): 

        print(f"\n  - {source}:\n")
        table_set = StoppingPowerTableSet.from_default_source(source)
        available_ions = table_set.get_available_ions()

        ## Display loaded tables
        print(f"  Loaded {len(sources)} tables.")
        print(f"  Available ions: {available_ions}")

        ## Plotting the stopping power curves (avoid built-in method to have more flexibility on plotting styles)
        for ion in available_ions:
            
            table = table_set.get(ion)
            stopping_power = table.stopping_power
            energy = table.energy
            ax.plot(energy, stopping_power, label=f"{ion}" if i==(len(sources)-1) else None,
                    linestyle=linestyles[i], marker=markers[i], color=table.color, alpha=0.3+0.3*i, linewidth=5-2*i)
        
        custom_lines.append(Line2D([0], [0], color="black", lw=3, linestyle=linestyles[i], marker=markers[i],
                                   label=f"{source}", markersize=8 if markers[i] else 0))
        

    # Plot settings
    ax.set_xlabel("Energy [MeV/u]")
    ax.set_ylabel("Stopping Power [MeV/cm]")
    ax.set_xscale("log")
    ax.grid(True, alpha=0.5)
    # Legend settings
    ax.legend()
    handles, labels = ax.get_legend_handles_labels()
    particle_legend = ax.legend(handles, labels, bbox_to_anchor=(0.775, 0.75), loc='upper left', borderaxespad=0.)
    source_legend = ax.legend(custom_lines, sources, bbox_to_anchor=(0.71, 0.98), loc='upper left', borderaxespad=0.)
    ax.add_artist(particle_legend)

    plt.show()

if __name__ == "__main__":
    main()
