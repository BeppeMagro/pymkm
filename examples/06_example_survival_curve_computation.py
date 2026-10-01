"""Example: compute survival curves with all pyMKM biological model families.

The example compares, at one selected LET, classic MKM, SMK, OSMK 2021,
OSMK 2023, and MCF-MKM. Each model is configured with its own published-style
HSG parameter set and evaluated through the common SFTable interface.
"""

import matplotlib.pyplot as plt

from pymkm.io.table_set import StoppingPowerTableSet
from pymkm.mktable.core import MKTable, MKTableParameters
from pymkm.sftable.core import SFTable, SFTableParameters


def main():
    atomic_number = 6  # C
    source = "mstar_3_12"
    model_name = "Kiefer-Chatterjee"
    core_type = "energy-dependent"
    let_value = 500.0  # MeV/cm = 50 keV/um

    mkm_parameters = {
        "domain_radius": 0.29,
        "nucleus_radius": 3.9,
        "alpha0": 0.150,
        "beta0": 0.0593,
    }

    smk_parameters = {
        "domain_radius": 0.28,
        "nucleus_radius": 8.1,
        "alpha0": 0.174,
        "beta0": 0.0568,
        "z0": 66.0,
    }

    osmk_parameters = {
        "domain_radius": 0.23,
        "nucleus_radius": 8.1,
        "alphaL": 0.0,
        "alphaS": 0.21,
        "beta0": 0.043,
        "z0": 88.0,
        "K": 3.0,
        "zR": 28.0,
        "gamma": 1.30,
        "Rm": 2.9,
        "f_rd_max": 3.00,
        "f_z0_max": 3.53,
        "Rmax": 4.46,
    }

    # HSG MCF-MKM parameters from Parisi et al. The canonical/default nucleus
    # treatment is the Parisi-consistent scaled formulation.
    mcf_parameters = {
        "domain_radius": 0.28,
        "nucleus_radius": 4.5,
        "alpha0": 0.188,
        "beta0": 0.057,
        "mcf_nucleus_mode": "scaled",
    }

    pO2 = 0.0  # full hypoxia [mmHg] for the OSMK examples

    print(
        f"Loading stopping-power table for Z={atomic_number} "
        f"from '{source}'..."
    )
    sp_table_set = (
        StoppingPowerTableSet.from_default_source(source)
        .filter_by_ions([atomic_number])
    )

    mk_table = MKTable(
        parameters=MKTableParameters(
            domain_radius=mkm_parameters["domain_radius"],
            nucleus_radius=mkm_parameters["nucleus_radius"],
            beta0=mkm_parameters["beta0"],
            model_name=model_name,
            core_radius_type=core_type,
        ),
        sp_table_set=sp_table_set,
    )

    smk_table = MKTable(
        parameters=MKTableParameters(
            domain_radius=smk_parameters["domain_radius"],
            nucleus_radius=smk_parameters["nucleus_radius"],
            z0=smk_parameters["z0"],
            model_name=model_name,
            core_radius_type=core_type,
            use_stochastic_model=True,
        ),
        sp_table_set=sp_table_set,
    )

    osmk_table = MKTable(
        parameters=MKTableParameters(
            domain_radius=osmk_parameters["domain_radius"],
            nucleus_radius=osmk_parameters["nucleus_radius"],
            z0=osmk_parameters["z0"],
            model_name=model_name,
            core_radius_type=core_type,
            use_stochastic_model=True,
        ),
        sp_table_set=sp_table_set,
    )

    mcf_table = MKTable(
        parameters=MKTableParameters(
            domain_radius=mcf_parameters["domain_radius"],
            nucleus_radius=mcf_parameters["nucleus_radius"],
            alpha0=mcf_parameters["alpha0"],
            beta0=mcf_parameters["beta0"],
            model_name=model_name,
            core_radius_type=core_type,
            use_mcf_model=True,
            mcf_nucleus_mode=mcf_parameters["mcf_nucleus_mode"],
        ),
        sp_table_set=sp_table_set,
    )

    common_osmk = {
        "mktable": osmk_table,
        "alphaS": osmk_parameters["alphaS"],
        "alphaL": osmk_parameters["alphaL"],
        "beta0": osmk_parameters["beta0"],
        "K": osmk_parameters["K"],
        "pO2": pO2,
    }

    model_configs = [
        {
            "label": "MKM",
            "model": "classic",
            "apply_oxygen_effect": False,
            "params": {
                "mktable": mk_table,
                "alpha0": mkm_parameters["alpha0"],
                "beta0": mkm_parameters["beta0"],
            },
        },
        {
            "label": "SMK",
            "model": "stochastic",
            "apply_oxygen_effect": False,
            "params": {
                "mktable": smk_table,
                "alpha0": smk_parameters["alpha0"],
                "beta0": smk_parameters["beta0"],
            },
        },
        {
            "label": "OSMK2021",
            "model": "stochastic",
            "apply_oxygen_effect": True,
            "params": {
                **common_osmk,
                "zR": osmk_parameters["zR"],
                "gamma": osmk_parameters["gamma"],
                "Rm": osmk_parameters["Rm"],
            },
        },
        {
            "label": "OSMK2023",
            "model": "stochastic",
            "apply_oxygen_effect": True,
            "params": {
                **common_osmk,
                "f_rd_max": osmk_parameters["f_rd_max"],
                "f_z0_max": osmk_parameters["f_z0_max"],
                "Rmax": osmk_parameters["Rmax"],
            },
        },
        {
            "label": "MCF-MKM",
            "model": "mcf",
            "apply_oxygen_effect": False,
            "params": {
                "mktable": mcf_table,
                "alpha0": mcf_parameters["alpha0"],
                "beta0": mcf_parameters["beta0"],
            },
        },
    ]

    survival_results = {}
    for config in model_configs:
        sf_table = SFTable(parameters=SFTableParameters(**config["params"]))
        sf_table.compute(
            ion=atomic_number,
            let=let_value,
            force_recompute=True,
            model=config["model"],
            apply_oxygen_effect=config["apply_oxygen_effect"],
        )
        survival_results[config["label"]] = sf_table.table[0]

    _, ax = plt.subplots(figsize=(8, 6))
    color = sp_table_set.get(atomic_number).color

    styles = {
        "MKM": {"linestyle": "-", "marker": None, "alpha": 0.45},
        "SMK": {"linestyle": "--", "marker": None, "alpha": 0.70},
        "OSMK2021": {"linestyle": ":", "marker": None, "alpha": 0.75},
        "OSMK2023": {"linestyle": "-.", "marker": None, "alpha": 0.85},
        "MCF-MKM": {"linestyle": "-", "marker": "o", "alpha": 1.0},
    }

    for label, result in survival_results.items():
        style = styles[label]
        ax.plot(
            result["data"]["dose"],
            result["data"]["survival_fraction"],
            label=label,
            color=color,
            linestyle=style["linestyle"],
            linewidth=3,
            alpha=style["alpha"],
            marker=style["marker"],
            markersize=4 if style["marker"] else 0,
        )

    ax.set_xlim(0, 8)
    ax.set_ylim(1e-4, 1)
    ax.set_yscale("log")
    ax.set_xlabel("Dose [Gy]")
    ax.set_ylabel("Survival fraction")
    ax.set_title(f"LET = {let_value / 10:.1f} keV/um")
    ax.grid(True, which="both", linestyle="--", alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
