"""Example: generate an MCF-MKM microdosimetric table.

The MCF-MKM table stores the two dose-averaged quantities required to recover
its LQ coefficients:

    c_bar       = dose-averaged non-Poisson correction factor [-]
    z_bar_c     = dose-averaged corrected domain specific energy [Gy]

The default ``mcf_nucleus_mode='scaled'`` is the formulation consistent with
the original Parisi MCF-MKM. The alternative ``'integrated'`` ATS treatment is
also available and can be selected explicitly.
"""

import matplotlib.pyplot as plt

from pymkm.io.table_set import StoppingPowerTableSet
from pymkm.mktable.core import MKTable, MKTableParameters


def main():
    cell_type = "V79"
    atomic_numbers = [1, 2, 6, 10]  # H, He, C, Ne
    source = "mstar_3_12"

    model_name = "Kiefer-Chatterjee"
    core_type = "energy-dependent"

    # V79 MCF-MKM parameters used for the average-trend calculations in
    # Parisi et al. (Phys. Med. Biol. 67, 185013, 2022).
    domain_radius = 0.26  # um
    nucleus_radius = 4.0  # um
    alpha0 = 0.125  # Gy^-1
    beta0 = 0.020  # Gy^-2
    alpha_ref = 0.190  # Gy^-1
    beta_ref = 0.020  # Gy^-2

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
        alpha0=alpha0,
        beta0=beta0,
        model_name=model_name,
        core_radius_type=core_type,
        use_mcf_model=True,
        mcf_nucleus_mode="scaled",  # canonical/default MCF formulation
    )

    print(f"Computing MCF-MKM table for Z={atomic_numbers}...")
    mcf_table = MKTable(parameters=params, sp_table_set=sp_table_set)
    mcf_table.compute(ions=atomic_numbers, parallel=True)

    # MCF-MKM has two native MKTable output quantities.
    _, axes = plt.subplots(nrows=1, ncols=2, figsize=(11, 5))
    mcf_table.plot(
        ions=atomic_numbers,
        x="energy",
        y="c_bar",
        verbose=True,
        ax=axes[0],
        show=False,
    )
    mcf_table.plot(
        ions=atomic_numbers,
        x="energy",
        y="z_bar_c",
        verbose=True,
        ax=axes[1],
        show=False,
    )
    plt.tight_layout()
    plt.show()

    output_path = "./MCF_table.txt"
    export_params = {
        "CellType": cell_type,
        "Alpha_ref": alpha_ref,
        "Beta_ref": beta_ref,
        "Alpha0": alpha0,
        "Beta0": beta0,
    }
    mcf_table.write_txt(
        model="mcf",
        params=export_params,
        filename=output_path,
        max_atomic_number=max(atomic_numbers),
    )
    print(f"MCF-MKM table written to: {output_path}")


if __name__ == "__main__":
    main()
