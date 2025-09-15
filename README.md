<p align="center">
  <img src="assets/pymkm_logo_none.png" alt="pyMKM logo" width="400"/>
</p>

# 🧬 pyMKM

[![CI](https://github.com/BeppeMagro/pymkm/actions/workflows/ci.yml/badge.svg)](https://github.com/BeppeMagro/pymkm/actions)
[![Docs](https://img.shields.io/badge/docs-online-success)](https://beppemagro.github.io/pymkm/)
[![Deploy Docs](https://github.com/BeppeMagro/pymkm/actions/workflows/gh-pages.yml/badge.svg)](https://github.com/BeppeMagro/pymkm/actions/workflows/gh-pages.yml)
[![TestPyPI](https://img.shields.io/badge/TestPyPI-pymkm-blue)](https://test.pypi.org/project/pymkm/)

**pyMKM** is an open-source Python package for the computation of microdosimetric quantities and cell survival predictions based on the **Microdosimetric Kinetic Model (MKM)**, its **stochastic extension (SMK)**, and its **oxygen-aware version (OSMK)**.

It is intended for use in radiobiology research, Monte Carlo-based dosimetry, and biologically guided treatment planning in hadrontherapy.

📘 **[Official Documentation](https://beppemagro.github.io/pymkm/)**  
📝 **[Paper (draft)](---)**

---

## 📦 Features

- 🔬 Full support for MKM, SMK, and OSMK (2021 & 2023 versions)
- 📈 Dose-averaged microdosimetric table generation
- 🎯 Event-by-event stochastic modeling of survival
- 🧪 Oxygen effect corrections with LET or event scaling
- 📐 Track structure models: Kiefer–Chatterjee and Scholz–Kraft
- 📊 Validated on 150+ datasets from H, He, C, Ne ions
- ⚙️ Modular architecture, parallelizable computation
- ✅ 100% test coverage, cross-version CI (Python 3.9–3.12)

---

## 📥 Installation

From PyPI (coming soon):

```bash
pip install pymkm
```

From source:

```bash
git clone https://github.com/BeppeMagro/pymkm.git
cd pymkm
pip install -e .[dev]
```

---

## 🧪 Quick Start

```python
from pymkm.mktable import MKTable, MKTableParameters

## Select input parameters for specific energy tables generation
atomic_numbers = [2, 6, 8] # He, C, O
source = "mstar_3_12" # Source code used to generate stopping power tables (available with pymkm: fluka_2020_0, geant4_11_3_0 or mstar_3_12)
domain_radius = 0.32 # μm
nucleus_radius = 3.9 # μm
alpha0 = 0.172 # 1/Gy
beta0 = 0.0615 # 1/Gy^2

## Load stopping power tables
sp_table_set = StoppingPowerTableSet.from_default_source(source).filter_by_ions(atomic_numbers)

## Store input parameters
params = MKTableParameters(
    domain_radius=domain_radius,
    nucleus_radius=nucleus_radius,
    beta0=beta0,
    )

## Generate specific energy table
mk_table = MKTable(parameters=params, sp_table_set=sp_table_set)
mk_table.compute(ions=atomic_numbers, parallel=True)

## Plot specific energies result using built-in method
mk_table.plot(ions=atomic_numbers, x="energy", y="z_bar_star_domain", verbose=True)

## Write the MKTable to a .txt file
path = "./MKM_table.txt"
params = {
    "CellType": cell_type,
    "Alpha_0": alpha0,
    "Beta": beta0
    }
mk_table.write_txt(params=params, filename=path)
```

More examples available in the `examples/` folder and documentation.

---

## 📂 Project Structure

```text
pymkm/
├── biology/        # OSMK oxygen effects and modulation
├── data/           # Ion tables and stopping powers
├── io/             # Data registry and loaders
├── mktable/        # MKM/SMK microdosimetric table computation
├── physics/        # Track structure and dose integration
├── sftable/        # Survival curve computations
├── utils/          # Geometry, interpolation, parallelism
tests/              # Unit and integration tests
examples/           # Demonstration scripts
validation_results/ # Validation datasets and figures
```

---

## 🧪 Testing

```bash
pytest
```

All modules are covered by tests and validated with published benchmark data.  
Continuous integration is provided via GitHub Actions.

---

## 📖 Citation

If you use `pyMKM` in your research, please cite:

> Magro G., Pavanello V., Jia Y., Grevillot L., Glimelius L. and Mairani A.  
> **pyMKM: An Open-Source Python Package for Microdosimetric Kinetic Model Calculation in Research and Clinical Applications**  
> GitHub: https://github.com/BeppeMagro/pymkm

---

## 📄 License

This project is licensed under the **MIT License** (for code) and **CC BY 4.0** (for scientific content).  
See the [LICENSE](LICENSE) file for more details.

---

## 💰 Funding

This work was funded by the National Plan for NRRP Complementary Investments (PNC) in the call for the funding of research initiatives for technologies and innovative trajectories in the health – project n. PNC0000003 – *AdvaNced Technologies for Human-centrEd Medicine* (project acronym: **ANTHEM** – Cascade Call launched by SPOKE 3 POLIMI: **PRECISION**).

---

## 🌐 Links

- 📘 Docs: [https://beppemagro.github.io/pymkm/](https://beppemagro.github.io/pymkm/)
- 🔬 Article: [---](---)
- 💬 Issues: [https://github.com/BeppeMagro/pymkm/issues](https://github.com/BeppeMagro/pymkm/issues)