"""Example: compare all stopping-power sources distributed with pyMKM."""

from itertools import cycle

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from pymkm.io.data_registry import get_available_sources
from pymkm.io.table_set import StoppingPowerTableSet


def main():
    sources = get_available_sources()
    print("Available default stopping-power sources:")
    for source in sources:
        print(f"  - {source}")

    _, ax = plt.subplots(figsize=(10, 6))

    # Cycle styles so that the example remains valid if additional default
    # stopping-power sources are added in future pyMKM releases.
    style_cycle = cycle([
        ("-", None),
        ("--", None),
        (":", "s"),
        ("-.", "o"),
    ])
    source_styles = {source: next(style_cycle) for source in sources}
    source_handles = []

    for source_index, source in enumerate(sources):
        table_set = StoppingPowerTableSet.from_default_source(source)
        available_ions = table_set.get_available_ions()
        linestyle, marker = source_styles[source]

        print(f"\n{source}")
        print(f"  Loaded {len(table_set)} ion tables.")
        print(f"  Available ions: {available_ions}")

        for ion in available_ions:
            table = table_set.get(ion)
            ax.plot(
                table.energy,
                table.stopping_power,
                label=ion if source_index == len(sources) - 1 else None,
                linestyle=linestyle,
                marker=marker,
                color=table.color,
                alpha=0.35 + 0.20 * source_index,
                linewidth=max(1.5, 4.5 - source_index),
                markersize=4 if marker else 0,
            )

        source_handles.append(
            Line2D(
                [0],
                [0],
                color="black",
                linewidth=2.5,
                linestyle=linestyle,
                marker=marker,
                markersize=7 if marker else 0,
                label=source,
            )
        )

    ax.set_xlabel("Energy [MeV/u]")
    ax.set_ylabel("Stopping power [MeV/cm]")
    ax.set_xscale("log")
    ax.grid(True, alpha=0.4)

    ion_handles, ion_labels = ax.get_legend_handles_labels()
    ion_legend = ax.legend(
        ion_handles,
        ion_labels,
        title="Ion",
        bbox_to_anchor=(1.01, 0.75),
        loc="upper left",
        borderaxespad=0.0,
    )
    source_legend = ax.legend(
        handles=source_handles,
        title="Source",
        bbox_to_anchor=(1.01, 1.0),
        loc="upper left",
        borderaxespad=0.0,
    )
    ax.add_artist(ion_legend)
    ax.add_artist(source_legend)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
