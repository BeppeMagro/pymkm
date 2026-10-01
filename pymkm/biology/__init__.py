# This file marks this directory as a Python package
"""
Biological effect models and correction tools.

This subpackage contains functions and models used to compute
biologically relevant quantities, including oxygen-effect corrections
for OSMK models and quantities required by the Mayo Clinic Florida
Microdosimetric Kinetic Model (MCF-MKM).

Modules
-------

- :mod:`mkm_model`:
  Provides utilities for computing the MKM and SMK linear-quadratic
  coefficients from dose-averaged microdosimetric quantities.

- :mod:`oxygen_effect`:
  Provides utilities for computing relative radioresistance (R),
  OSMK 2023 scaling factors (f_rd, f_z0), and oxygen-corrected α and β.

- :mod:`mcf_model`:
  Provides utilities for computing the MCF-MKM non-Poisson correction
  factor, the scaled single-event specific energy of the nucleus, the
  dose-averaged quantities c_bar and z_bar_c, and the resulting
  MCF-MKM linear-quadratic coefficients.

These functions are used by higher-level pyMKM components when the
corresponding biological model or correction is enabled.
"""
