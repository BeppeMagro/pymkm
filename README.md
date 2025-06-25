# pyMKM

[![CI](https://github.com/BeppeMagro/pymkm/actions/workflows/ci.yml/badge.svg)](https://github.com/BeppeMagro/pymkm/actions)
[![TestPyPI](https://img.shields.io/badge/TestPyPI-pymkm-blue)](https://test.pypi.org/project/pymkm/)

`pyMKM` is an open-source Python package for computing microdosimetric quantities using MKM, SMK, and OSMK models.  
It is designed for use in radiobiology research and clinical applications in hadrontherapy.

---

## 📦 Features

- Dose and LET-based microdosimetric modeling
- MKM, SMK, and OSMK formulations
- Support for amorphous track structure models (Kiefer-Chatterjee, Scholz-Kraft)
- Automatic table generation for survival calculation
- Full test coverage with `pytest`

---

## 📥 Installation

From PyPI:

```bash
pip install pymkm
```

From source:

```bash
git clone https://github.com/BeppMagro/pymkm.git
cd pymkm
pip install -e .[dev]
```

---

## 🧪 Testing

To run tests and measure code coverage:

```bash
run_tests.bat
```

---

## 📂 Project structure

```text
pymkm/                  # Main package
  ├── biology/          # Oxygen effects and biological corrections
  ├── data/             # Stopping power and ion tables
  ├── io/               # Data registry and file loading
  ├── mktable/          # MKM and SMK table generation
  ├── physics/          # Particle tracks and specific energy
  ├── sftable/          # Survival fraction calculations
  ├── utils/            # Geometry tools, interpolation, parallelism
tests/                  # Full test suite (not installed)
examples/               # Usage examples (not installed)
validation_results/     # Model benchmarks (not installed)
```

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 📖 Citation

If you use this software in your research, please cite:

> Magro G., Pavanello V., Mairani A.  
> *pyMKM: Microdosimetric Kinetic Model utilities for radiobiology research and hadrontherapy.*

> GitHub repository: [pymkm](https://github.com/BeppeMagro/pymkm)

---

## 💰 Funding

This work was funded by the National Plan for NRRP Complementary Investments (PNC) in the call for the funding of research initiatives for technologies and innovative trajectories in the health – project n. PNC0000003 – *AdvaNced Technologies for Human-centrEd Medicine* (project acronym: **ANTHEM** – Cascade Call launched by SPOKE 3 POLIMI: “PRECISION”).

---

## 🌐 Links

- 📘 Documentation: *(coming soon)*
- 🧪 Test results: via GitHub Actions
- 💬 Issues: [GitHub Issues](https://github.com/BeppeMagro/pymkm/issues)