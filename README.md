<p align="center">
  <img src="https://raw.githubusercontent.com/BeppeMagro/pymkm/main/assets/pymkm_logo_none.png" alt="pyMKM logo" width="400"/>
</p>

---

# 🧬 pyMKM

[![CI](https://github.com/BeppeMagro/pymkm/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/BeppeMagro/pymkm/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-online-success)](https://beppemagro.github.io/pymkm/)
[![Deploy Docs](https://github.com/BeppeMagro/pymkm/actions/workflows/gh-pages.yml/badge.svg)](https://github.com/BeppeMagro/pymkm/actions/workflows/gh-pages.yml)
[![TestPyPI](https://img.shields.io/badge/TestPyPI-pymkm-blue)](https://test.pypi.org/project/pymkm/)

**pyMKM** is an open-source Python package for microdosimetric calculations and cell-survival modelling based on the **Microdosimetric Kinetic Model (MKM)** and related formulations, including the **stochastic MKM (SMK)**, oxygen-modified **OSMK** models, and the **MCF-MKM** formulation based on impact-parameter-dependent specific energy.

It is intended for radiobiology research, Monte Carlo-based dosimetry, and biologically guided treatment planning in hadrontherapy.

📘 **[Official Documentation](https://beppemagro.github.io/pymkm/)**
📝 **[Paper (Computation, 2025)](https://doi.org/10.3390/computation13110264)**

---

## 📦 Features

- 🔬 MKM, SMK, OSMK (2021 and 2023), and MCF-MKM calculations
- 📈 Dose-averaged microdosimetric table generation
- 🎯 Cell-survival calculations using linear-quadratic model coefficients
- 🧪 Oxygen-effect corrections with LET- or event-based scaling
- 📐 Kiefer–Chatterjee and Scholz–Kraft radial dose models
- 🧬 MCF-MKM impact-parameter averaging with `scaled` nucleus mode by default and optional `integrated` mode
- 📊 Validation workflows against published benchmark data for H, He, C, and Ne ions
- ⚙️ Modular architecture with optional parallel computation
- ✅ Automated tests and reproducible documentation/build utilities

---

## 📥 Installation

For the stable release available on PyPI:

```bash
pip install pymkm
```

For the beta release available on TestPyPI:

```bash
pip install -i https://test.pypi.org/simple/ pymkm
```

From source:

```bash
git clone https://github.com/BeppeMagro/pymkm.git
cd pymkm
pip install -e ".[dev]"
```

---

## 🧪 Quick Start

The main high-level interfaces are exported directly by `pymkm`.
The example below generates a classical MKM microdosimetric table for He, C, and O ions using bundled MSTAR stopping-power data.

```python
from pymkm import MKTable, MKTableParameters, StoppingPowerTableSet

# Ions and stopping-power source
atomic_numbers = [2, 6, 8]  # He, C, O
source = "mstar_3_12"

sp_table_set = (
    StoppingPowerTableSet
    .from_default_source(source)
    .filter_by_ions(atomic_numbers)
)

# Classical MKM configuration
params = MKTableParameters(
    domain_radius=0.32,   # µm
    nucleus_radius=3.9,   # µm
    beta0=0.0615,         # Gy^-2
)

# Generate the microdosimetric table
mk_table = MKTable(parameters=params, sp_table_set=sp_table_set)
mk_table.compute(ions=atomic_numbers, parallel=True)

# Plot dose-averaged specific energy
mk_table.plot(
    ions=atomic_numbers,
    x="energy",
    y="z_bar_star_domain",
    verbose=True,
)

# Export a classical MKM table
mk_table.write_txt(
    params={
        "CellType": "HSG",
        "Alpha_0": 0.172,
        "Beta": 0.0615,
    },
    filename="MKM_table.txt",
    model="classic",
)
```

MCF-MKM is enabled through `MKTableParameters`:

```python
mcf_params = MKTableParameters(
    domain_radius=0.28,
    nucleus_radius=4.5,
    alpha0=0.188,
    beta0=0.057,
    use_mcf_model=True,
    mcf_nucleus_mode="scaled",  # default; "integrated" is also available
)
```

Additional workflows, including SMK, OSMK, survival calculations, MCF-MKM, plotting, and validation examples, are available in the `examples/` and `validation_results/` directories.

---

## 📂 Project Structure

```text
pymkm/
├── biology/        # MKM/SMK/MCF and oxygen-effect biological models
├── data/           # Bundled stopping-power datasets and element metadata
├── io/             # Data registry, stopping-power tables, and loaders
├── mktable/        # MKM/SMK/MCF microdosimetric table computation
├── physics/        # Track-structure and specific-energy calculations
├── sftable/        # Survival-fraction calculations
└── utils/          # Geometry, integration, interpolation, and parallel tools

docs/               # Sphinx documentation sources
examples/           # Demonstration scripts
tests/              # Unit and integration tests
tools/              # Development, documentation, build, and Git helpers
validation_results/ # Published-data validation workflows and outputs
```

---

## 🧪 Testing

Run the repository test helper:

```bash
python tools/run_tests.py
```

or run pytest directly:

```bash
pytest
```

The repository also contains validation workflows based on published benchmark data. Continuous integration is provided through GitHub Actions.

---

## 📚 Documentation

Regenerate the API reference and build the Sphinx HTML documentation with:

```bash
python tools/build_docs.py
```

The generated documentation is written to `docs/build/html/` and the public documentation is deployed through GitHub Pages.

---

## 📖 Citation

If you use `pyMKM` in your research, please cite:

> Magro, G., Pavanello, V., Jia, Y., Grevillot, L., Glimelius, L., & Mairani, A. (2025).
> **pyMKM: An Open-Source Python Package for Microdosimetric Kinetic Model Calculation in Research and Clinical Applications.**
> *Computation*, 13(11), 264. https://doi.org/10.3390/computation13110264

---

## 📄 License

This project is licensed under the **MIT License** for code and **CC BY 4.0** for scientific content.
See the [LICENSE](LICENSE) file for more details.

---

## 💰 Funding

This work was funded by the National Plan for NRRP Complementary Investments (PNC) in the call for the funding of research initiatives for technologies and innovative trajectories in the health – project n. PNC0000003 – *AdvaNced Technologies for Human-centrEd Medicine* (project acronym: **ANTHEM** – Cascade Call launched by SPOKE 3 POLIMI: **PRECISION**).

---

## 🌐 Links

- 📘 Docs: [https://beppemagro.github.io/pymkm/](https://beppemagro.github.io/pymkm/)
- 🔬 Article: [https://doi.org/10.3390/computation13110264](https://doi.org/10.3390/computation13110264)
- 💬 Issues: [https://github.com/BeppeMagro/pymkm/issues](https://github.com/BeppeMagro/pymkm/issues)
