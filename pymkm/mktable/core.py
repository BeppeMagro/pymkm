"""
Core classes for MKM and SMK microdosimetric table generation.

This module defines:
- :class:`MKTableParameters`: configuration container for geometry, models, and computation settings
- :class:`MKTable`: main interface for generating, storing, and exporting microdosimetric tables

Supports both classic MKM and stochastic SMK models, with optional OSMK 2023 corrections for hypoxia.
Each MKTable instance manages the full computation pipeline per ion type, including saving,
loading, displaying, and exporting results.
"""

from dataclasses import asdict
from typing import Optional, Union, Literal
from dataclasses import dataclass
from tabulate import tabulate
from pathlib import Path
import warnings
import pickle
import datetime
import pandas as pd
from pymkm.io.table_set import StoppingPowerTableSet
from pymkm.utils.geometry_tools import GeometryTools


@dataclass
class MKTableParameters:
    """
    Configuration container for MKTable model and geometry parameters.
    
    This dataclass defines the physical, numerical, and model-specific parameters
    needed to generate microdosimetric tables using MKM or SMK.
    
    :ivar domain_radius: Radius of the sensitive domain (μm).
    :ivar nucleus_radius: Radius of the cell nucleus (μm).
    :ivar z0: Saturation parameter z₀ (Gy). Required for SMK.
    :ivar beta0: LQ model quadratic coefficient β₀ (Gy⁻²). Required for MKM.
    :ivar model_name: Track structure model: 'Kiefer-Chatterjee' or 'Scholz-Kraft'.
    :ivar core_radius_type: Core radius model: 'constant' or 'energy-dependent'.
    :ivar base_points_b: Number of impact parameter sampling points.
    :ivar base_points_r: Number of radial sampling points.
    :ivar use_stochastic_model: If True, enables SMK computation.
    :ivar pO2: Oxygen partial pressure (mmHg).
    :ivar f_rd_max: Max scaling factor for domain radius under hypoxia.
    :ivar f_z0_max: Max scaling factor for z₀ under hypoxia.
    :ivar Rmax: Maximum radioresistance ratio at 0 mmHg pO2.
    :ivar K: Half-effect oxygen pressure (mmHg).
    :ivar apply_oxygen_effect: Enable OSMK 2023 correction if True.
    """

    @classmethod
    def from_dict(cls, config: dict) -> "MKTableParameters":
        """
        Create an MKTableParameters instance from a dictionary.
        
        :param config: Dictionary of configuration fields.
        :type config: dict
        
        :returns: Populated MKTableParameters instance.
        :rtype: MKTableParameters
        
        :raises ValueError: If unknown keys are present in the dictionary.
        """
        valid_keys = set(cls.__dataclass_fields__.keys())
        incoming_keys = set(config.keys())
        extra_keys = incoming_keys - valid_keys
    
        if extra_keys:
            raise ValueError(
                f"Unrecognized keys in MKTableParameters config: {sorted(extra_keys)}"
            )
    
        return cls(**config)

    domain_radius: float
    nucleus_radius: float
    z0: Optional[float] = None
    beta0: Optional[float] = None

    model_name: str = "Kiefer-Chatterjee"
    core_radius_type: str = "energy-dependent"
    base_points_b: int = GeometryTools.generate_default_radii.__defaults__[1]
    base_points_r: int = GeometryTools.generate_default_radii.__defaults__[1]

    use_stochastic_model: bool = False
    
    # --- OSMK 2023 Correction Parameters (optional) ---
    pO2: Optional[float] = None        # Oxygen partial pressure [mmHg]
    f_rd_max: Optional[float] = None   # Max scaling for domain radius
    f_z0_max: Optional[float] = None   # Max scaling for z0
    Rmax: Optional[float] = None       # Maximum R at pO2 = 0
    K: float = 3.0                     # Half-effect oxygen pressure [mmHg]
    
    apply_oxygen_effect: bool = False  # Enables correction if True and parameters are present


class MKTable:
    """
    Main handler for microdosimetric table generation using MKM or SMK.
    
    This class manages the physical model, geometry, table computation,
    result storage, and export functionalities.
    """
    def __repr__(self):
        return (f"<MKTable model={self.model_version}, r_d={self.params.domain_radius}, "
                f"R_n={self.params.nucleus_radius}>")

    @property
    def model_version(self) -> str:
        """
        Get the active model version as a string label.
        
        :returns: 'stochastic' if SMK is enabled, otherwise 'classic' (MKM).
        :rtype: str
        """
        return "stochastic" if self.params.use_stochastic_model else "classic"
    
    def _default_filename(self, extension: str = ".pkl") -> Path:
        """
        Generate a default filename based on model and geometry settings.
    
        :param extension: File extension (e.g., '.pkl' or '.txt').
        :type extension: str
    
        :returns: Path object pointing to default output location.
        :rtype: Path
        """
        root = Path.home() / ".pyMKM" / extension.strip(".")
        root.mkdir(parents=True, exist_ok=True)
        suffix = extension if extension.startswith(".") else f".{extension}"

        s = self.sp_table_set.source_info.replace(" ", "_").replace("/", "-")
        r_d = f"rd{self.params.domain_radius:.2f}"
        r_n = f"rn{self.params.nucleus_radius:.1f}"
        z0 = f"z0{self.params.z0:.1f}" if self.params.z0 is not None else "z0None"
        b0 = f"b0{self.params.beta0:.4f}" if self.params.beta0 is not None else "b0None"

        prefix = "smk" if self.params.use_stochastic_model else "mkm"
        model_abbr = "kc" if self.params.model_name.lower().startswith("kiefer") else "sk"
        core_abbr = "const" if self.params.core_radius_type == "constant" else "ed"
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{prefix}_{s}_{model_abbr}_{core_abbr}_rd{r_d}_Rn{r_n}_z0{z0}_b0{b0}_{timestamp}{suffix}"
        return root / filename
    
    def save(self, filename: Optional[Union[str, Path]] = None):
        """
        Save the computed MKTable results to a pickle file.
    
        :param filename: Optional output file path. If None, uses default name.
        :type filename: str or Path, optional
    
        :raises ValueError: If no results have been computed.
        """
        if not self.table:
            raise ValueError("Cannot save: MKTable has not been computed yet. Run 'compute()' first.")
        path = Path(filename) if filename else self._default_filename(".pkl")
        with open(path, "wb") as f:
            pickle.dump(self.table, f)
        print(f"✅ Table saved to: {path}")
        
    def load(self, filename: Union[str, Path]):
        """
        Load previously saved MKTable results from a pickle file.
    
        :param filename: Path to the .pkl file containing saved table data.
        :type filename: str or Path
    
        :raises FileNotFoundError: If the specified file does not exist.
        """
        path = Path(filename)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        with open(path, "rb") as f:
            self.table = pickle.load(f)
        print(f"📂 Table loaded from: {path}")

    def summary(self, verbose: bool = False):
        """
        Print a summary of the current MKTable configuration.
    
        Displays model type, physical parameters, and sampling settings.
        If verbose is True, lists available ions and technical details.
    
        :param verbose: If True, include detailed configuration info.
        :type verbose: bool, optional
        """
        param_dict = asdict(self.params)
    
        # Main physical parameters with symbols and units
        main_parameters = [
            ("r_d [μm]", param_dict["domain_radius"]),
            ("R_n [μm]", param_dict["nucleus_radius"]),
            ("z₀ [Gy]", param_dict["z0"]),
            ("β₀ [Gy⁻²]", param_dict["beta0"]),
        ]
    
        # Technical parameters
        technical_parameters = [
            ("Track structure model", param_dict["model_name"]),
            ("Track core type", param_dict["core_radius_type"]),
            ("Sampling points for b", param_dict["base_points_b"]),
            ("Sampling points for r", param_dict["base_points_r"]),
        ]
    
        print("\nMKTable Configuration")
        print(f"\nModel version: {self.model_version}")
        print(f"\nStopping power source: {self.sp_table_set.source_info}")
    
        print(tabulate(main_parameters, headers=["Parameter", "Value"], tablefmt="fancy_grid"))
    
        if verbose:
            ions = self.sp_table_set.get_available_ions()
            if ions:
                print("\nAvailable ions:")
                print(", ".join(ions))
            print()
            print(tabulate(technical_parameters, headers=["Setting", "Value"], tablefmt="fancy_grid"))
            print("\nNote: Sampling points refer to base values before internal refinement.")


    def __init__(self, parameters: MKTableParameters, sp_table_set: Optional[StoppingPowerTableSet] = None):
        """
        Initialize an MKTable instance with model configuration and stopping power data.
    
        :param parameters: Geometry, model, and numerical settings for table generation.
        :type parameters: MKTableParameters
        :param sp_table_set: Optional stopping power data. If None, a default set is loaded.
        :type sp_table_set: Optional[StoppingPowerTableSet]
        """
        self.params = parameters
        self.sp_table_set = sp_table_set or StoppingPowerTableSet.from_default_source("fluka_2020_0")
        self.table = {}
        self._validate_parameters()

    def _validate_parameters(self):
        """
        Perform internal consistency checks on MKTableParameters.
        
        Validates:
        - Presence of either z₀ or β₀ depending on MKM/SMK usage
        - Correct handling of z₀ and β₀ interaction
        - Presence of OSMK parameters if oxygen effect correction is enabled
        
        :raises ValueError: If critical parameters are missing or incompatible.
        :raises Warning: If redundant or conflicting values are detected (e.g., both z₀ and β₀).
        """
        p = self.params

        # --- Base validation: MKM / SMK logic ---
        if p.z0 is None and p.beta0 is None:
            raise ValueError(
                "Both z0 and beta0 are missing. MKM typically requires beta0; SMK typically requires z0.\n"
                "At least one must be provided to proceed."
            )

        if p.use_stochastic_model:
            if p.z0 is None and p.beta0 is not None:
                warnings.warn("z0 not provided. It will be derived from beta0, which is not standard for SMK.")
            if p.z0 is not None and p.beta0 is not None:
                warnings.warn("Both z0 and beta0 provided. z0 will be used for SMK; beta0 retained only for post-processing.")
        else:
            if p.z0 is not None and p.beta0 is None:
                warnings.warn("z0 provided but beta0 is missing. Will use z0 directly, which is not standard for MKM.")
            if p.z0 is not None and p.beta0 is not None:
                warnings.warn("Both z0 and beta0 provided. In MKM, beta0 will be used to compute z0; the user-provided z0 will be discarded.")
                p.z0 = None # force recalculation from beta0
        
        # --- OSMK 2023 correction validation ---
        if p.apply_oxygen_effect:
            if not p.use_stochastic_model:
                raise ValueError("apply_oxygen_effect=True requires use_stochastic_model=True.")
            required_osmk = ("pO2", "f_rd_max", "f_z0_max", "Rmax")
            missing = [k for k in required_osmk if not hasattr(p, k) or getattr(p, k) is None]
            if missing:
                raise ValueError(f"apply_oxygen_effect=True but missing OSMK 2023 parameters: {missing}")

    def _refresh_parameters(self, original_params: Optional[MKTableParameters] = None) -> None:
        """
        Refresh internal parameters and print changes from previous configuration.
    
        Used after modifying or reloading MKTableParameters.
    
        :param original_params: Optional previous version for comparison.
        :type original_params: Optional[MKTableParameters]
        """
        updated_fields = {}
        current = self.params
        updated = MKTableParameters.from_dict(asdict(current))
    
        reference = original_params or updated  # fallback: compare to self (no-op)
    
        for field_name in current.__dataclass_fields__:
            old_value = getattr(reference, field_name)
            new_value = getattr(updated, field_name)
            if old_value != new_value:
                setattr(self.params, field_name, new_value)
                updated_fields[field_name] = (old_value, new_value)
    
        if updated_fields:
            print("\nMKTableParameters updated:")
            for k, (old, new) in updated_fields.items():
                print(f" - {k}: {old} → {new}")
            print("\nUpdated configuration summary:")
            self.summary(verbose=False)

    def get_table(self, ion: Union[str, int]) -> pd.DataFrame:
        """
        Retrieve computed table for a specific ion.
    
        :param ion: Ion identifier (name, symbol, or atomic number).
        :type ion: str or int
    
        :returns: Microdosimetric table as a DataFrame.
        :rtype: pandas.DataFrame
    
        :raises ValueError: If results are not available or ion is not found.
        """
        if not self.table:
            raise ValueError("No computed results found. Run 'compute()' first.")

        ion_key = self.sp_table_set._map_to_fullname(ion)

        if ion_key not in self.table:
            raise ValueError(f"Ion '{ion}' not found in computed table.")

        return self.table[ion_key]["data"]

    def display(self, preview_rows: int = 5):
        """
        Print a formatted preview of the computed tables for all ions.
    
        Shows metadata, model parameters, and head/tail of each ion's table.
    
        :param preview_rows: Number of rows to display from start and end of each table.
        :type preview_rows: int
    
        :raises ValueError: If no tables have been computed.
        """
        if not self.table:
            raise ValueError("No computed results found. Please run 'compute()' first.")
    
        print("\n📊 Computed Microdosimetric Tables:")
        for ion_key, result in self.table.items():
            print(f"\n🔹 Ion: {ion_key}")
            
            sp_info_table = [(k, v) for k, v in result["stopping_power_info"].items()]
            print(tabulate(sp_info_table, headers=["Stopping Power Info", "Value"], tablefmt="grid"))

            param_table = [(k, v) for k, v in result["params"].items()]
            print(tabulate(param_table, headers=["Parameter", "Value"], tablefmt="grid"))
       
            df = result["data"]
            top = df.head(preview_rows)
            bottom = df.tail(preview_rows)
            
            print(f"\nTop {preview_rows} rows:")
            print(tabulate(top, headers="keys", tablefmt="fancy_grid", showindex=False))
    
            print(f"\nBottom {preview_rows} rows:")
            print(tabulate(bottom, headers="keys", tablefmt="fancy_grid", showindex=False))
            print("-" * 60)
    
    def write_txt(
        self,
        *,
        params: dict,
        filename: Union[str, Path] = None,
        model: Literal["classic", "stochastic"] = None,
        max_atomic_number: Optional[int] = None
    ):
        """
        Export results to a .txt file compatible with external tools.
        
        Required `params` depend on the selected model:
        
        For model="classic" (MKM):
            Required:
                - "CellType": str
                - "Alpha_0": float
            Optional:
                - "Beta": float
        
        For model="stochastic" (SMK):
            Required:
                - "CellType": str
                - "Alpha_ref": float
                - "Beta_ref": float
                - "Alpha0": float
            Optional:
                - "Beta0": float
                - "scale_factor": float (defaults to 1.0)
        
        :param params: Model-dependent metadata to include in the header.
        :type params: dict
        :param filename: Output file path. If None, a default name is generated.
        :type filename: str or Path, optional
        :param model: Force output format. If None, inferred from configuration.
        :type model: Literal["classic", "stochastic"], optional
        :param max_atomic_number: Maximum Z for ions to include. 
            If None (default), all available ions are included.
        :type max_atomic_number: int, optional
        
        :raises ValueError:
            - If no data has been computed yet.
            - If required parameters are missing.
            - If atomic number exceeds available Z range.
            - If Beta/Beta0 consistency checks fail.
        :raises KeyError: 
            - If unexpected or invalid keys are present in `params`.
            - If required microdosimetric columns are missing in the table.
        """
        
        if not self.table:
            raise ValueError("Cannot write: MKTable has not been computed yet. Run 'compute()' first.")
    
        model = model or self.model_version
    
        if model == "stochastic" and not self.params.use_stochastic_model:
            raise ValueError("Stochastic output requested but MKTable was computed in classic mode.")
    
        if model == "classic":
            allowed_keys = {"CellType", "Alpha_0", "Beta"}
            required_keys = {"CellType", "Alpha_0"}
        else:
            allowed_keys = {"CellType", "Alpha_ref", "Beta_ref", "scale_factor", "Alpha0", "Beta0"}
            required_keys = {"CellType", "Alpha_ref", "Beta_ref", "Alpha0"}
    
        incoming_keys = set(params.keys())
    
        if not required_keys.issubset(incoming_keys):
            missing = required_keys - incoming_keys
            raise KeyError(f"Missing required keys in 'params': {missing}")
    
        extra = incoming_keys - allowed_keys
        if extra:
            raise KeyError(f"Unexpected keys in 'params': {extra}")
        
        # Determine maximum available Z from stopping_power_info
        available_Z = [self.table[k]["stopping_power_info"]["atomic_number"] for k in self.table]
        max_Z_table = max(available_Z)
    
        # If no max_atomic_number is provided, include all ions
        if max_atomic_number is None:
            max_atomic_number = max_Z_table
        elif max_atomic_number > max_Z_table:
            raise ValueError(f"Requested max_atomic_number={max_atomic_number} exceeds computed table max Z={max_Z_table}.")
    
        # Common header
        path = Path(filename) if filename else self._default_filename(".txt")
        path.parent.mkdir(parents=True, exist_ok=True)
    
        with open(path, "w") as f:
            f.write(f"CellType  {params['CellType']}\n\n")
    
            if model == "classic":
                f.write(f"Parameter Alpha_0 {params['Alpha_0']:.4f}\n")
                beta_param = params.get("Beta")
                beta_obj = self.params.beta0
    
                if beta_obj is None and beta_param is None:
                    raise ValueError("Beta must be defined either in params or in self.params.beta0.")
                if beta_obj is not None and beta_param is not None:
                    if abs(beta_obj - beta_param) > 1e-6:
                        raise ValueError(
                            f"Mismatch between beta0 in params ({beta_param}) and self.params ({beta_obj})"
                        )
                beta = beta_obj if beta_obj is not None else beta_param
                f.write(f"Parameter Beta {beta:.4f}\n")
                f.write(f"Parameter DomainRadius {self.params.domain_radius:.4f}\n")
                f.write(f"Parameter NucleusRadius {self.params.nucleus_radius:.4f}\n\n")
    
            else:  # stochastic
                f.write(f"Parameter Alpha_ref {params['Alpha_ref']:.4f}\n")
                f.write(f"Parameter Beta_ref {params['Beta_ref']:.4f}\n")
                scale = params.get("scale_factor", 1.0)
                if 'scale_factor' not in params:
                    warnings.warn("'scale_factor' not provided, defaulting to 1.00")
                f.write(f"Parameter scale_factor {scale:.2f}\n")
                f.write(f"Parameter Alpha0 {params['Alpha0']:.4f}\n")
    
                beta_param = params.get("Beta0")
                beta_obj = self.params.beta0
                if beta_obj is None and beta_param is None:
                    raise ValueError("Beta0 must be defined either in params or in self.params.beta0.")
                if beta_obj is not None and beta_param is not None:
                    if abs(beta_obj - beta_param) > 1e-6:
                        raise ValueError(
                            f"Mismatch between Beta0 in params ({beta_param}) and self.params ({beta_obj})"
                        )
                beta = beta_obj if beta_obj is not None else beta_param
                f.write(f"Parameter Beta0 {beta:.4f}\n\n")
    
            for ion_key, result in self.table.items():
                Z = self.table[ion_key]["stopping_power_info"].get("atomic_number")
                if Z > max_atomic_number:
                    continue
                
                symbol = self.table[ion_key]["stopping_power_info"].get("ion_symbol")
                df = result["data"]
                f.write(f"Fragment {symbol}\n")
    
                if model == "classic":
                    if "z_bar_star_domain" not in df.columns:
                        raise KeyError(f"Missing expected column 'z_bar_star_domain' for ion {ion_key}.")
                    for _, row in df.iterrows():
                        f.write(f"{row['energy']:.5e} {row['z_bar_star_domain']:.5e}\n")
                else:
                    expected_cols = ["z_bar_domain", "z_bar_star_domain", "z_bar_nucleus"]
                    for col in expected_cols:
                        if col not in df.columns:
                            raise KeyError(f"Missing expected column '{col}' for ion {ion_key}.")
                    for _, row in df.iterrows():
                        f.write(
                            f"{row['energy']:.5e} {row['z_bar_domain']:.5e} {row['z_bar_star_domain']:.5e} {row['z_bar_nucleus']:.5e}\n"
                        )
                f.write("\n")
    
        print(f"\nTable written to: {path}")
    
        
        
        
