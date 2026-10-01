"""
Plotting utilities for MKTable results.

This module defines the :meth:`MKTable.plot` method for visualizing model-specific
microdosimetric quantities as functions of energy or LET. Supported outputs include
MKM/SMK specific-energy quantities and MCF-MKM dose-averaged correction quantities.
"""

from typing import List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np

from .core import MKTable


plt.rcParams.update({
    "axes.linewidth": 1.2,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "xtick.color": "black",
    "ytick.color": "black",
    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,
    "xtick.major.size": 5,
    "ytick.major.size": 5,
    "legend.fontsize": 14,
    "axes.titlesize": 12,
})


_MODEL_Y_COLUMNS = {
    "classic": {"z_bar_star_domain"},
    "stochastic": {"z_bar_star_domain", "z_bar_domain", "z_bar_nucleus"},
    "mcf": {"c_bar", "z_bar_c"},
}

_Y_LABELS = {
    "z_bar_star_domain": r"$\bar{z}^{*}$ [Gy]",
    "z_bar_domain": r"$\bar{z}_d$ [Gy]",
    "z_bar_nucleus": r"$\bar{z}_n$ [Gy]",
    "c_bar": r"$\bar{c}$ [-]",
    "z_bar_c": r"$\bar{z}^{(c)}$ [Gy]",
}


def _default_y_column(model_version: str) -> str:
    """Return the default y-axis quantity for the active model."""
    return "c_bar" if model_version == "mcf" else "z_bar_star_domain"


def _validate_plot_columns(x: str, y: str, model_version: str) -> None:
    """
    Validate plot-axis columns for the active MKTable model.

    :param x: Name of the x-axis variable ('energy' or 'let').
    :type x: str
    :param y: Name of the y-axis variable.
    :type y: str
    :param model_version: Active model ('classic', 'stochastic', or 'mcf').
    :type model_version: str

    :raises ValueError: If x, y, or model_version is invalid.
    """
    allowed_x = {"energy", "let"}
    if x not in allowed_x:
        raise ValueError(f"Invalid x-axis: '{x}'. Allowed values are: {sorted(allowed_x)}")

    if model_version not in _MODEL_Y_COLUMNS:
        raise ValueError(
            f"Unsupported MKTable model version: '{model_version}'. "
            f"Allowed values are: {sorted(_MODEL_Y_COLUMNS)}"
        )

    allowed_y = _MODEL_Y_COLUMNS[model_version]
    if y not in allowed_y:
        raise ValueError(
            f"Invalid y-axis: '{y}' for model '{model_version}'. "
            f"Allowed values are: {sorted(allowed_y)}"
        )


def plot(
    self: MKTable,
    ions: Optional[List[Union[str, int]]] = None,
    *,
    x: str = "energy",
    y: Optional[str] = None,
    verbose: bool = False,
    ax: Optional[plt.Axes] = None,
    show: Optional[bool] = True,
    title: Optional[bool] = False,
):
    """
    Plot model-specific quantities from the MKTable.

    For classic MKM, the available y quantity is ``z_bar_star_domain``.
    For stochastic SMK, ``z_bar_domain`` and ``z_bar_nucleus`` are additionally
    available. For MCF-MKM, the available quantities are ``c_bar`` and ``z_bar_c``.

    If ``y`` is omitted, the default is ``z_bar_star_domain`` for MKM/SMK and
    ``c_bar`` for MCF-MKM.

    :param ions: List of ions to plot. If None, all computed ions are used.
    :type ions: list[str or int], optional
    :param x: x-axis variable ('energy' or 'let').
    :type x: str
    :param y: Model-specific y-axis variable. If None, uses the model default.
    :type y: str, optional
    :param verbose: Show model configuration in the plot.
    :type verbose: bool
    :param ax: Matplotlib Axes object to draw on. If None, a new figure is created.
    :type ax: Optional[matplotlib.axes.Axes]
    :param show: If True, displays the plot. Set False when embedding or scripting.
    :type show: Optional[bool]
    :param title: If True, displays a title with stopping-power and track-model information.
    :type title: Optional[bool]

    :raises RuntimeError: If table is empty.
    :raises ValueError: If x or y are invalid for the active model.
    """
    if not self.table:
        raise RuntimeError("No computed results found. Run `compute()` before plotting.")

    model_version = self.model_version
    y = _default_y_column(model_version) if y is None else y
    _validate_plot_columns(x, y, model_version)

    ions = ions or self.sp_table_set.get_available_ions()
    ions = [self.sp_table_set._map_to_fullname(ion) for ion in ions]

    x_label_map = {"energy": "Energy [MeV/u]", "let": "LET [MeV/cm]"}

    created_fig = False
    if ax is None:
        _, ax = plt.subplots()
        created_fig = True

    x_min, x_max, y_max = np.inf, -np.inf, -np.inf
    for ion in ions:
        df = self.table[ion]["data"]
        x_vals = df[x].values
        y_vals = df[y].values
        x_min = min(x_min, x_vals.min())
        x_max = max(x_max, x_vals.max())
        y_max = max(y_max, y_vals.max())

    for ion in ions:
        df = self.table[ion]["data"]
        ion_symbol = self.sp_table_set.get(ion).ion_symbol
        color = self.sp_table_set.get(ion).color
        ax.plot(df[x], df[y], label=ion_symbol, color=color, alpha=0.5, linewidth=6)

    ax.set_xlabel(x_label_map[x])
    ax.set_ylabel(_Y_LABELS[y])

    if title:
        plot_title = (
            f"Source: {self.sp_table_set.source_info}, "
            f"Track model: {self.params.model_name} (Core: {self.params.core_radius_type})"
        )
        ax.set_title(plot_title, wrap=True)

    ax.set_xscale("log" if x == "energy" else "linear")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0, y_max * 1.05)
    ax.grid(True, which="both", linestyle="--", alpha=0.1)
    ax.legend()

    if verbose:
        param_dict = vars(self.params)
        main_parameters = [
            (r"$r_d$", param_dict["domain_radius"], "μm"),
            (r"$R_n$", param_dict["nucleus_radius"], "μm"),
        ]

        if model_version == "mcf":
            main_parameters.extend([
                (r"$\alpha_0$", param_dict["alpha0"], "Gy⁻¹"),
                (r"$\beta_0$", param_dict["beta0"], "Gy⁻²"),
            ])
        else:
            if param_dict.get("z0") is not None:
                main_parameters.append((r"$z_0$", param_dict["z0"], "Gy"))
            if param_dict.get("beta0") is not None:
                main_parameters.append((r"$\beta_0$", param_dict["beta0"], "Gy⁻²"))

        info_lines = [f"Model: {model_version}"] + [
            f"{key}: {value:.3f} {unit}" for key, value, unit in main_parameters
        ]
        info_text = "\n".join(info_lines)

        ax.text(
            0.05,
            0.05,
            info_text,
            transform=ax.transAxes,
            fontsize=14,
            verticalalignment="bottom",
            horizontalalignment="left",
            bbox=dict(
                facecolor="white",
                alpha=0.85,
                edgecolor="black",
                boxstyle="round",
            ),
        )

    if show and created_fig:
        plt.tight_layout()
        plt.show()


MKTable.plot = plot
