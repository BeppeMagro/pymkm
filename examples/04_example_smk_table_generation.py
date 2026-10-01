"""Example: generate a stochastic-MKM (SMK) microdosimetric table.

The script demonstrates how to:
  - Load and filter a stopping-power source.
  - Configure a stochastic MKTable.
  - Compute z_bar*, z_bar_d, and z_bar_n.
  - Plot the three SMK quantities.
  - Export the result using the stochastic text format.
"""

import matplotlib.pyplot as plt

from pymkm.io.table_set import StoppingPowerTableSet
from pymkm.mktable.core import MKTable, MKTableParameters


def main():
    cell_type = "HSG"
    atomic_numbers = [2, 6, 8]  # He, C, O
    source = "mstar_3_12"

    model_name = "Kiefer-Chatterjee"
    core_type = "energy-dependent"
    domain_radius = 0.23  # um
    nucleus_radius = 8.1  # um
    z0 = 88.0  # Gy

    # Biological parameters used when the table is exported or later consumed
    # by an SFTable. They are deliberately kept separate from the purely
    # microdosimetric SMK table configuration.
    alpha0 = 0.12  # Gy^-1
    beta0 = 0.043  # Gy^-2
    alpha_ref = 0.12  # Gy^-1
    beta_ref = 0.0615  # Gy^-2
    clinical_scale_factor = 1.0

    print(
        f"Loading stopping-power tables for Z={atomic_numbers} "
        f"from '{source}'..."
    )
    sp_table_set = (
        StoppingPowerTableSet.from_default_source(source)
        .filter_by_ions(atomic_numbers)
    )

    # For SMK, z0 is the microdosimetric saturation parameter. beta0 is not
    # required to generate the table when z0 is supplied explicitly.
    params = MKTableParameters(
        domain_radius=domain_radius,
        nucleus_radius=nucleus_radius,
        z0=z0,
        model_name=model_name,
        core_radius_type=core_type,
        use_stochastic_model=True,
    )

    print(f"Computing stochastic MKM table for Z={atomic_numbers}...")
    smk_table = MKTable(parameters=params, sp_table_set=sp_table_set)
    smk_table.compute(ions=atomic_numbers, parallel=True)

    _, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
    smk_table.plot(
        ions=atomic_numbers,
        x="energy",
        y="z_bar_star_domain",
        verbose=True,
        ax=axes[0],
        show=False,
    )
    smk_table.plot(
        ions=atomic_numbers,
        x="energy",
        y="z_bar_domain",
        verbose=True,
        ax=axes[1],
        show=False,
    )
    smk_table.plot(
        ions=atomic_numbers,
        x="energy",
        y="z_bar_nucleus",
        verbose=True,
        ax=axes[2],
        show=False,
    )
    plt.tight_layout()
    plt.show()

    output_path = "./SMK_table.txt"
    export_params = {
        "CellType": cell_type,
        "Alpha_ref": alpha_ref,
        "Beta_ref": beta_ref,
        "scale_factor": clinical_scale_factor,
        "Alpha0": alpha0,
        "Beta0": beta0,
    }
    smk_table.write_txt(
        model="stochastic",
        params=export_params,
        filename=output_path,
        max_atomic_number=max(atomic_numbers),
    )
    print(f"SMK table written to: {output_path}")


if __name__ == "__main__":
    main()
