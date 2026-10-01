"""
Core classes for survival fraction (SF) table generation.

This module defines:

- :class:`SFTableParameters`: A dataclass storing all parameters required to compute
  survival fraction curves using MKM, SMK, OSMK, or MCF-MKM models.
- :class:`SFTable`: A computation manager that integrates MKTable results with
  biological model parameters to produce survival fraction outputs.

The module supports OSMK 2021 and OSMK 2023 models, as well as MCF-MKM survival calculations based on precomputed MCF quantities from MKTable.
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import warnings
from tabulate import tabulate

from pymkm.mktable.core import MKTable

@dataclass
class SFTableParameters:
    """
    Configuration container for computing survival fraction (SF) curves using MKM, SMK, OSMK, or MCF-MKM models.

    :param mktable: Precomputed MKTable containing specific energy values.
    :type mktable: pymkm.mktable.core.MKTable

    :param alpha0: Total linear coefficient α₀ in the LQ model [Gy⁻¹]. Required for normoxic MKM/SMK survival calculations. For OSMK, at least two of `alpha0`, `alphaL`, and `alphaS` are required. For MCF-MKM, if omitted, it is retrieved from `mktable.params` and must match the value used to generate the MKTable.
    :type alpha0: Optional[float]

    :param beta0: Quadratic coefficient β₀ in the LQ model [Gy⁻²]. If not provided, it is retrieved from `mktable.params`.
    :type beta0: Optional[float]

    :param dose_grid: Dose grid [Gy] over which to compute survival fractions. Defaults to np.arange(0, 15.5, 0.5).
    :type dose_grid: Optional[np.ndarray]

    :param alphaL: Linear coefficient for lethal lesions [Gy⁻¹] (OSMK 2021/2023).
    :type alphaL: Optional[float]

    :param alphaS: Linear coefficient for sublethal lesions [Gy⁻¹] (OSMK 2021/2023).
    :type alphaS: Optional[float]

    :param zR: Radiation quality–dependent oxygen parameter [Gy] (OSMK 2021 only).
    :type zR: Optional[float]

    :param gamma: Exponent for R_max(zd) expression (OSMK 2021 only).
    :type gamma: Optional[float]

    :param Rm: Minimum value of R_max (OSMK 2021 only).
    :type Rm: Optional[float]

    :param f_rd_max: Maximum domain radius scaling factor under hypoxia (OSMK 2023 only).
    :type f_rd_max: Optional[float]

    :param f_z0_max: Maximum saturation parameter scaling factor under hypoxia (OSMK 2023 only).
    :type f_z0_max: Optional[float]

    :param Rmax: Maximum radioresistance at pO₂ = 0 mmHg (OSMK 2023 only).
    :type Rmax: Optional[float]

    :param K: Oxygen pressure [mmHg] at which R(pO₂) = (1 + Rmax)/2. Default is 3.0 (Inaniwa 2021).
    :type K: Optional[float]

    :param pO2: Oxygen partial pressure [mmHg] at which to evaluate the oxygen effect.
        Enables OSMK mode if set.
    :type pO2: Optional[float]
    """

    mktable: MKTable
    alpha0: Optional[float] = None
    beta0: Optional[float] = None
    dose_grid: np.ndarray = field(default_factory=lambda: np.arange(0, 15.5, 0.5))

    # Optional parameters for OSMK
    alphaL: Optional[float] = None
    alphaS: Optional[float] = None
    zR: Optional[float] = None
    gamma: Optional[float] = None
    Rm: Optional[float] = None
    f_rd_max: Optional[float] = None
    f_z0_max: Optional[float] = None
    Rmax: Optional[float] = None
    K: float = 3.0  # mmHg, default
    pO2: Optional[float] = None  # mmHg

    def __post_init__(self):
        """
        Validate parameter consistency and derive missing values if necessary.
        
        - Ensures `mktable` is an instance of MKTable.
        - Checks and fills in missing `beta0` from the MKTable.
        - For MCF-MKM, checks and fills in missing `alpha0` from the MKTable.
        - Requires `alpha0` for normoxic MKM/SMK survival calculations.
        - Enforces consistency between `alpha0`, `alphaL`, and `alphaS` when pO2 is set.
        - Prevents mixing of OSMK 2021 and OSMK 2023 parameter sets.
        
        :raises TypeError: If `mktable` is not a MKTable instance.
        :raises ValueError: If required parameters are missing or inconsistent.
        """
        if not isinstance(self.mktable, MKTable):
            raise TypeError("mktable must be an instance of MKTable")

        if not isinstance(self.dose_grid, np.ndarray):
            self.dose_grid = np.array(self.dose_grid, dtype=float)

        # === Validate beta0 ===
        beta_from_table = self.mktable.params.beta0
        if self.beta0 is None:
            if beta_from_table is not None:
                self.beta0 = beta_from_table
                warnings.warn("beta0 not provided, using value from MKTable.params.")
            else:
                raise ValueError("beta0 must be provided either explicitly or via MKTable.params.")
        elif beta_from_table is not None and abs(beta_from_table - self.beta0) > 1e-6:
            raise ValueError(
                f"Mismatch between provided beta0 ({self.beta0}) and MKTable.params.beta0 ({beta_from_table})."
            )
    
        # === Validate alpha0 for MCF-MKM ===
        if self.mktable.model_version == "mcf":
            if self.pO2 is not None:
                raise ValueError("OSMK oxygen-effect parameters are not supported for MCF-MKM.")

            alpha_from_table = self.mktable.params.alpha0
            if self.alpha0 is None:
                if alpha_from_table is not None:
                    self.alpha0 = alpha_from_table
                    warnings.warn("alpha0 not provided, using value from MKTable.params.")
                else:
                    raise ValueError("alpha0 must be provided either explicitly or via MKTable.params for MCF-MKM.")
            elif alpha_from_table is not None and abs(alpha_from_table - self.alpha0) > 1e-6:
                raise ValueError(
                    f"Mismatch between provided alpha0 ({self.alpha0}) and MKTable.params.alpha0 ({alpha_from_table})."
                )

        # === Require alpha0 for normoxic MKM/SMK ===
        if self.mktable.model_version != "mcf" and self.pO2 is None and self.alpha0 is None:
            raise ValueError(
                "alpha0 must be provided for normoxic MKM/SMK survival calculations."
            )

        # === Handle alpha0, alphaL, alphaS consistency only if OSMK is requested ===
        if self.pO2 is not None:
            alphaL, alphaS, alpha0 = self.alphaL, self.alphaS, self.alpha0
            n_provided = sum(v is not None for v in (alpha0, alphaL, alphaS))

            if n_provided < 2:
                raise ValueError("For OSMK (pO2 specified), at least two of alpha0, alphaL, and alphaS must be provided.")

            if alpha0 is None:
                self.alpha0 = alphaL + alphaS
            elif alphaL is None:
                self.alphaL = alpha0 - alphaS
            elif alphaS is None:
                self.alphaS = alpha0 - alphaL
            else:
                if not np.isclose(alphaL + alphaS, alpha0, atol=1e-6):
                    raise ValueError(f"Inconsistent values for OSMK: alpha0={alpha0}, alphaL + alphaS = {alphaL + alphaS}")

            # === Ensure exclusivity between OSMK 2021 and 2023 parameter sets ===
            has_osmk2021 = any(x is not None for x in (self.zR, self.gamma, self.Rm))
            has_osmk2023 = any(x is not None for x in (self.f_rd_max, self.f_z0_max, self.Rmax))
            if has_osmk2021 and has_osmk2023:
                raise ValueError("Cannot mix OSMK 2021 (zR, γ, Rm) and OSMK 2023 (f_rd_max, f_z0_max, Rmax) parameters.")
                
                
    @classmethod
    def from_dict(cls, config: dict) -> "SFTableParameters":
        """
        Create an SFTableParameters instance from a dictionary.
    
        Unrecognized keys in the dictionary will trigger a warning.
    
        :param config: Dictionary of parameters with keys matching the dataclass fields.
        :type config: dict
    
        :returns: A populated SFTableParameters instance.
        :rtype: SFTableParameters
    
        :raises ValueError: If unknown keys are present in the configuration dictionary.
        """
        valid_keys = set(cls.__dataclass_fields__.keys())
        incoming_keys = set(config.keys())
        extra_keys = incoming_keys - valid_keys

        if extra_keys:
            raise ValueError(
                f"Unrecognized keys in SFTableParameters config: {sorted(extra_keys)}"
            )

        return cls(**config)


class SFTable:
    def __init__(self, parameters: SFTableParameters):
        """
        Initialize the SFTable with a set of biological and model parameters.
        
        :param parameters: An SFTableParameters instance containing model and oxygen settings.
        :type parameters: SFTableParameters
        """
        self.params = parameters
        self.table = None

    def __repr__(self):
        if self.params.alphaL is not None and self.params.alphaS is not None:
            alpha0 = self.params.alphaL + self.params.alphaS
        else:
            alpha0 = self.params.alpha0
        beta0 = self.params.beta0
        return f"<SFTable | α_0 = {alpha0}, β_0 = {beta0}>"

    def summary(self):
        """
        Print a detailed summary of the current survival model configuration.
        
        Displays LQ parameters and OSMK-related settings if applicable.
        """
        print("\nSFTable Configuration")
        table = [
            ("\u03b1_0 [Gy^-1]", f"{self.params.alpha0:.3f}" if self.params.alpha0 is not None else "None"),
            ("\u03b2_0 [Gy^-2]", f"{self.params.beta0:.3f}" if self.params.beta0 is not None else "None"),
        ]

        if self.params.pO2 is not None:
            table += [
                ("pO2 [mmHg]", f"{self.params.pO2:.2f}"),
                ("α_L [Gy^-1]", f"{self.params.alphaL:.3f}" if self.params.alphaL is not None else "None"),
                ("α_S [Gy^-1]", f"{self.params.alphaS:.3f}" if self.params.alphaS is not None else "None"),
                ("zR (2021) [Gy]", f"{self.params.zR:.2f}" if self.params.zR is not None else "None"),
                ("γ (2021)", f"{self.params.gamma:.2f}" if self.params.gamma is not None else "None"),
                ("Rm (2021)", f"{self.params.Rm:.2f}" if self.params.Rm is not None else "None"),
                ("f_rd_max (2023)", f"{self.params.f_rd_max:.2f}" if self.params.f_rd_max is not None else "None"),
                ("f_z0_max (2023)", f"{self.params.f_z0_max:.2f}" if self.params.f_z0_max is not None else "None"),
                ("Rmax (2023)", f"{self.params.Rmax:.2f}" if self.params.Rmax is not None else "None")
            ]

        print(tabulate(table, headers=["Parameter", "Value"], tablefmt="fancy_grid"))

    def display(self, results: list):
        """
        Display the computed survival fraction results in a tabular format.
        
        :param results: List of dictionaries with keys 'params', 'calculation_info', and 'data'.
        :type results: list[dict]
        
        :raises ValueError: If no results are provided.
        """
        if not results:
            raise ValueError("No results to display. Please run 'compute()' first.")
    
        print("\n📈 Survival Fraction Results:")
        for idx, result in enumerate(results):
            params = result.get("params", {})
            calc_info = result.get("calculation_info", "N/A")
            df = result.get("data")
    
            print(f"\n🔹 Result {idx + 1}")
            print(tabulate(params.items(), headers=["Parameter", "Value"], tablefmt="grid"))
            print(f"\nCalculation Info: {calc_info}")
    
            if df is not None and not df.empty:
                print("\nData Table:")
                print(tabulate(df, headers="keys", tablefmt="fancy_grid", showindex=False))
            else:
                print("No data found in this result.")
            print("-" * 60)
