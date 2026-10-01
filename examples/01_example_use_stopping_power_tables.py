"""Example: load, interpolate, plot, and serialize stopping-power tables.

This script demonstrates how to:
  - Load a default stopping-power source and select one ion.
  - Resample the table on a custom energy grid.
  - Interpolate LET from energy and energy from LET.
  - Plot the table together with the interpolated points.
  - Serialize the filtered table set to JSON and load it back.
"""

import numpy as np
import matplotlib.pyplot as plt

from pymkm.io.table_set import StoppingPowerTableSet


def main():
    ion = "Carbon"
    source = "fluka_2020_0"

    print(f"Loading stopping-power tables from '{source}'...")
    table_set = StoppingPowerTableSet.from_default_source(source)
    table_set = table_set.filter_by_ions([ion])
    table = table_set.get(ion)

    print(f"Available ions after filtering: {table_set.get_available_ions()}")

    # Resample on a denser logarithmic energy grid within the original range.
    min_energy = table.energy.min()
    max_energy = table.energy.max()
    new_grid = np.logspace(np.log10(min_energy), np.log10(max_energy), 200)
    table.resample(new_grid)

    # Interpolate LET values at selected energies.
    selected_energies = np.logspace(
        np.log10(table.energy.min()),
        np.log10(table.energy.max()),
        4,
    )
    interpolated_let = table.interpolate(energy=selected_energies)

    # Interpolate energy values corresponding to selected LET values. Because the
    # stopping-power curve is not generally one-to-one, one LET may map to more
    # than one energy; interpolate(let=...) therefore returns a dictionary.
    selected_let = np.linspace(
        table.stopping_power.min() * 1.20,
        table.stopping_power.max() * 0.90,
        5,
    )
    interpolated_energy = table.interpolate(let=selected_let)

    # Plot the stopping-power table and interpolation examples.
    _, ax = plt.subplots()
    table.plot(show=False, ax=ax)

    ax.scatter(
        selected_energies,
        interpolated_let,
        facecolors="none",
        edgecolors=table.color,
        s=100,
        linewidth=1.8,
        label="Interpolated LET points",
    )

    first = True
    for let_value, energies in interpolated_energy.items():
        ax.scatter(
            energies,
            np.full_like(energies, let_value),
            marker="s",
            facecolors="none",
            edgecolors=table.color,
            s=100,
            linewidths=1.8,
            label="Interpolated energy points" if first else None,
        )
        first = False

    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys())
    ax.grid(True)
    plt.tight_layout()
    plt.show()

    # Serialize the filtered table set and demonstrate how to load it back.
    json_filename = f"fluka_stopping_power_table_{ion}.json"
    table_set.save(json_filename)
    print(f"Saved table set to: {json_filename}")

    reloaded_set = StoppingPowerTableSet.load(json_filename)
    print("Reloaded ions:", reloaded_set.get_available_ions())


if __name__ == "__main__":
    main()
