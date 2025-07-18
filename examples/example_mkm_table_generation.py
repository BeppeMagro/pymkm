import numpy as np
import matplotlib.pyplot as plt
from pymkm.mktable.core import MKTableParameters, MKTable
from pymkm.io.table_set import StoppingPowerTableSet

def main():

    # Select input parameters for rbe tables generation
    cell_type = "T"
    atomic_numbers = [2, 6, 8] # He, C, O
    source = "mstar_3_12" # Source code used to generate stopping power tables (available with pymkm: fluka_2020_0, geant4_11_3_0 or mstar_3_12)
    model_name = "Kiefer-Chatterjee" # Amorphous track structure model (Kiefer-Chatterjee or Scholz-Kraft)
    core_type = "energy-dependent" # Core radius model ('constant' or 'energy-dependent')
    domain_radius = 0.32 # μm
    nucleus_radius = 3.9 # μm
    alpha0 = 0.12 # 1/Gy
    beta0 = 0.0615 # 1/Gy^2

    print(f"\nGenerating stopping power tables for ion Z = {atomic_numbers} (using source '{source}')...")
    sp_table_set = StoppingPowerTableSet.from_default_source(source).filter_by_ions(atomic_numbers)

    # Store input parameters
    params = MKTableParameters(
        domain_radius=domain_radius,
        nucleus_radius=nucleus_radius,
        beta0=beta0,
        model_name=model_name,
        core_radius_type=core_type,
    )

    # Generate rbe table
    print(f"\nGenerating MKM tables for ion Z = {atomic_numbers} (using source '{source}')...")
    mk_table = MKTable(parameters=params, sp_table_set=sp_table_set)
    mk_table.compute(ions=atomic_numbers, parallel=True)

    # Plot model result using built-in method
    mk_table.plot(ions=atomic_numbers, x="energy", y="z_bar_star_domain", verbose=True)

    # Write to .txt file 
    path = "./MKM_table.txt"
    params = {
        "CellType": cell_type,
        "Alpha_0": alpha0,
        "Beta": beta0
    }
    mk_table.write_txt(params=params, filename=path, max_atomic_number=max(atomic_numbers))


if __name__ == "__main__":
    main()
