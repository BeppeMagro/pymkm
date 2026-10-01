"""Example: calculate an OSMK-2021 oxygen-enhancement-ratio trend.

The oxygen model acts during survival calculation; it does not define a new
microdosimetric table type. This example therefore creates an SMK-compatible
MKTable configuration and directly evaluates survival curves over the
stopping-power energy/LET grid for normoxic and hypoxic conditions.
"""

from concurrent.futures import ProcessPoolExecutor
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from pymkm.io.table_set import StoppingPowerTableSet
from pymkm.mktable.core import MKTable, MKTableParameters
from pymkm.sftable.core import SFTable, SFTableParameters
from pymkm.utils.parallel import optimal_worker_count


def compute_survival_curve(
    sf_table: SFTable,
    atomic_number: int,
    energy: float,
    let: float,
):
    """Compute one oxygen-corrected survival curve for an energy/LET pair."""
    sf_table.compute(
        ion=atomic_number,
        energy=energy,
        let=let,
        force_recompute=True,
        model="stochastic",
        apply_oxygen_effect=True,
    )

    entry = sf_table.table[0]
    return (
        energy,
        let,
        entry["data"]["dose"],
        entry["data"]["survival_fraction"],
    )


def inverse_dose_from_survival(
    doses: np.ndarray,
    survivals: np.ndarray,
    target_survival: float,
) -> float:
    """Interpolate the dose corresponding to a target survival fraction."""
    doses = np.asarray(doses, dtype=float)
    survivals = np.asarray(survivals, dtype=float)

    if not (0 < target_survival <= 1):
        raise ValueError("target_survival must be in the interval (0, 1].")
    if not np.all(np.diff(doses) > 0):
        raise ValueError("Dose values must be strictly increasing.")
    if not np.all(np.diff(survivals) < 0):
        raise ValueError("Survival curve must be strictly decreasing.")

    return float(np.interp(target_survival, survivals[::-1], doses[::-1]))


def main():
    atomic_number = 6  # C
    source = "mstar_3_12"
    model_name = "Kiefer-Chatterjee"
    core_type = "energy-dependent"

    domain_radius = 0.23  # um
    nucleus_radius = 8.1  # um
    z0 = 88.0  # Gy

    alphaL = 0.0  # Gy^-1
    alphaS = 0.21  # Gy^-1
    beta0 = 0.043  # Gy^-2
    K = 3.0  # mmHg
    zR = 28.0  # Gy
    gamma = 1.30
    Rm = 2.9

    oxygen_levels = {
        "hypoxia": 0.0,
        "normoxia": 160.0,
    }

    print(
        f"Loading stopping-power table for Z={atomic_number} "
        f"from '{source}'..."
    )
    sp_table_set = (
        StoppingPowerTableSet.from_default_source(source)
        .filter_by_ions([atomic_number])
    )
    sp_table = sp_table_set.get(atomic_number)

    energy = sp_table.energy
    let_values = sp_table.let
    energy_let_pairs = list(zip(energy, let_values))
    worker_count = optimal_worker_count(energy_let_pairs)

    # z0 is sufficient for the underlying stochastic microdosimetric table.
    mk_params = MKTableParameters(
        domain_radius=domain_radius,
        nucleus_radius=nucleus_radius,
        z0=z0,
        model_name=model_name,
        core_radius_type=core_type,
        use_stochastic_model=True,
    )
    osmk_table = MKTable(parameters=mk_params, sp_table_set=sp_table_set)

    dose_at_10_percent = {}

    for status, pO2 in oxygen_levels.items():
        sf_params = SFTableParameters(
            mktable=osmk_table,
            beta0=beta0,
            alphaS=alphaS,
            alphaL=alphaL,
            pO2=pO2,
            K=K,
            zR=zR,
            gamma=gamma,
            Rm=Rm,
        )
        sf_table = SFTable(parameters=sf_params)

        print(
            f"Calculating survival curves for Z={atomic_number}, "
            f"pO2={pO2:.1f} mmHg..."
        )
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            results = list(
                tqdm(
                    executor.map(
                        partial(compute_survival_curve, sf_table, atomic_number),
                        *zip(*energy_let_pairs),
                    ),
                    total=len(energy_let_pairs),
                    unit="energy",
                )
            )

        dose_at_10_percent[status] = np.array([
            inverse_dose_from_survival(dose, survival, target_survival=0.1)
            for _, _, dose, survival in results
        ])

    oer = dose_at_10_percent["hypoxia"] / dose_at_10_percent["normoxia"]

    _, ax = plt.subplots()
    ax.plot(
        let_values,
        oer,
        label=f"OER10 (pO2={oxygen_levels['hypoxia']:.1f} mmHg)",
        color=sp_table.color,
        linewidth=4,
        alpha=0.6,
    )
    ax.set_xscale("log")
    ax.set_xlim(1e2, 1e4)
    ax.set_ylim(1.0, float(np.max(oer)) * 1.1)
    ax.set_xlabel("LET [MeV/cm]")
    ax.set_ylabel("OER at 10% survival")
    ax.grid(alpha=0.4)
    ax.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
