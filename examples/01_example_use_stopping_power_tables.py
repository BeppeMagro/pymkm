import numpy as np
import matplotlib.pyplot as plt
from pymkm.io.table_set import StoppingPowerTableSet

"""
Example usage of StoppingPowerTableSet to visualize and manipulate stopping power tables.

This script demonstrates how to:
  - Load stopping power table from the default Fluka source ("fluka_2020_0") for one ion (C).
  - Resample the table on a new energy grid and interpolate specific points 
    (either stopping power form energy or energy from stopping power).
  - Plot the stopping power curve with interpolated points.
  - Serialize the table to a JSON file (bonus: load it back).
"""

def main():

    ## Load stopping power table
    ion = "Carbon"
    source = "fluka_2020_0" # Source code used to generate stopping power tables (available with pymkm: fluka_2020_0, geant4_11_3_0 or mstar_3_12)
    print(f"Loading stopping power tables from default Fluka source '{source}'...")
    table_set = StoppingPowerTableSet.from_default_source(source)
    # Filter stopping power tables based on ions of interest (filter_by_ions can take either atomic numbers (e.g. 6), symbols (e.g. "C") or full names (e.g. "Carbon"))
    print(f"\nLoaded!\nFiltering tabler for ion: {ion}")
    table_set = table_set.filter_by_ions([ion])
    # Transform table_set to a single table (StoppingPowerTable) for the ion of interest
    table = table_set.get(ion)

    ## Resample stopping power table
    print("\nResampling tables on a new energy grid...")
    new_grid = np.logspace(-1, 2, 200)
    min_energy = table.energy.min()
    max_energy = table.energy.max()
    table.resample(new_grid)

    ## Interpolate stopping power values at specific energies
    print("\nInterpolating new stopping power points...")
    min_energy = table.energy.min()
    max_energy = table.energy.max()
    new_energy_range = np.logspace(np.log10(min_energy), np.log10(max_energy), 4)
    stopping_power_values = table.interpolate(energy=new_energy_range)

    ## Interapolate energy for specific stopping power values
    # Some stopping power values may be associated with multiple energies: the interapolate method
    # will thus return a dictionary with stopping power values as keys and arrays of energies as values.
    print("\nInterpolating new energy points...")
    min_stopping_power = table.stopping_power.min()
    max_stopping_power = table.stopping_power.max()
    new_stopping_power_range = np.linspace(min_stopping_power * 1.20, max_stopping_power * 0.9, 5)
    energy_values = table.interpolate(let=new_stopping_power_range)

    ## Plot stopping power curve
    print("\nPlotting...")
    _, ax = plt.subplots()
    table.plot(show=False, ax=ax)

    # Overlay interpolated points (energy->stopping power)
    ax.scatter(
        new_energy_range, stopping_power_values,
        facecolors='none', edgecolors=table.color, s=100, linewidth=1.8,
        label="Interpolated SP points"
    )

    # Overlay interpolated points (stopping power->energy)
    isfirstInterapolation = False # To avoid duplicate legend entries
    for sp_val, energies in energy_values.items():
        ax.scatter(
            energies, np.full_like(energies, sp_val),
            marker='s', facecolor='none', edgecolor=table.color, s=100, linewidths=1.8,
            label="Interpolated energy points" if isfirstInterapolation else None
        )
        isfirstInterapolation = True

    # Deduplicate legend
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys())
       
    ax.grid(True)
    plt.tight_layout()
    plt.show()

    ## Serialize the table to a JSON file
    json_filename = f"fluka_stopping_power_table_{ion}.json"
    table_set.save(json_filename)
    print(f"\n\nSaved table set to JSON file: {json_filename}")
    # And eventually load it back
    # reloaded_set = StoppingPowerTableSet.load(json_filename)
    # print("Reloaded table set from JSON. Available ions:", reloaded_set.get_available_ions())

if __name__ == "__main__":
    main()
