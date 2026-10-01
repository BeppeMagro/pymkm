"""Example: generate a classic modified-MKM microdosimetric table.

The script demonstrates how to:
  - Load and filter a stopping-power source.
  - Configure a classic MKTable.
  - Compute saturation-corrected dose-mean specific energy z_bar*.
  - Plot the table for several ions.
  - Export the result using the classic MKM text format.
"""

from pymkm.io.table_set import StoppingPowerTableSet
from pymkm.mktable.core import MKTable, MKTableParameters


def main():
    cell_type = "HSG"
    atomic_numbers = [2, 6, 8]  # He, C, O
    source = "mstar_3_12"

    model_name = "Kiefer-Chatterjee"
    core_type = "energy-dependent"
    domain_radius = 0.32  # um
    nucleus_radius = 3.9  # um
    alpha0 = 0.172  # Gy^-1; used in the exported biological metadata
    beta0 = 0.0615  # Gy^-2

    print(
        f"Loading stopping-power tables for Z={atomic_numbers} "
        f"from '{source}'..."
    )
    sp_table_set = (
        StoppingPowerTableSet.from_default_source(source)
        .filter_by_ions(atomic_numbers)
    )

    params = MKTableParameters(
        domain_radius=domain_radius,
        nucleus_radius=nucleus_radius,
        beta0=beta0,
        model_name=model_name,
        core_radius_type=core_type,
    )

    print(f"Computing classic MKM table for Z={atomic_numbers}...")
    mk_table = MKTable(parameters=params, sp_table_set=sp_table_set)
    mk_table.compute(ions=atomic_numbers, parallel=True)

    mk_table.plot(
        ions=atomic_numbers,
        x="energy",
        y="z_bar_star_domain",
        verbose=True,
    )

    output_path = "./MKM_table.mkm"
    export_params = {
        "CellType": cell_type,
        "Alpha_0": alpha0,
        "Beta": beta0,
    }
    mk_table.write_txt(
        model="classic",
        params=export_params,
        filename=output_path,
        max_atomic_number=max(atomic_numbers),
    )
    print(f"Classic MKM table written to: {output_path}")


if __name__ == "__main__":
    main()
