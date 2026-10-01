# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- MCF-MKM microdosimetric table and cell-survival calculations.
- MCF nucleus-specific-energy modes: `scaled` (default) and optional `integrated`.
- Validation workflows for MCF biological weighting functions, RBE, and survival benchmarks.
- Automated Sphinx API generation and documentation build utilities.
- Cross-platform development helpers for testing, documentation, build, and release verification.

### Changed
- Refactored MKM, SMK, OSMK, and MCF biological calculations into explicit computation APIs and shared numerical utilities.
- Improved MKTable and SFTable configuration, persistence, validation, plotting, export, and reuse of precomputed results.
- Updated examples, package docstrings, README, and Sphinx documentation for MCF-MKM.
- Simplified dependency and packaging workflows with dedicated `docs` and `release` extras.
- Reduced source-distribution contents by keeping validation datasets and tests in the repository rather than the package distribution.
- Reorganized repository development and Git helper tools under `tools/`.

### Fixed
- Corrected MKTable default filename generation and sanitization.
- Corrected SFTable reuse behavior when `force_recompute=False`.
- Removed obsolete or misleading warnings and stale documentation references.

## [1.0.0] - 2025-11-10

### Added
- First stable release of pyMKM.
- MKM, SMK, and oxygen-modified MKM calculation workflows.
- Microdosimetric table generation and cell-survival calculations.
- Bundled stopping-power datasets and validation workflows.
- Public Python API, examples, automated tests, and Sphinx documentation.

## [0.1.0] - 2025-10-02

### Added
- First official public beta release of pyMKM.
