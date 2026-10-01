import numpy as np
import pytest

from pymkm.biology.mcf_model import (
    compute_scaled_nuclear_specific_energy,
    compute_mcf_correction_factor,
    compute_mcf_averaged_quantities,
    compute_mcf_lq_coefficients,
)

_trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


def test_compute_scaled_nuclear_specific_energy():
    z_domain = np.array([4.0, 2.0, 1.0])
    rd = 0.5
    Rn = 5.0

    result = compute_scaled_nuclear_specific_energy(z_domain, rd, Rn)
    expected = z_domain * (rd / Rn) ** 2

    assert np.allclose(result, expected)


def test_compute_scaled_nuclear_specific_energy_requires_positive_radii():
    z_domain = np.array([1.0, 2.0])

    with pytest.raises(ValueError, match="must be positive"):
        compute_scaled_nuclear_specific_energy(z_domain, 0.0, 5.0)

    with pytest.raises(ValueError, match="must be positive"):
        compute_scaled_nuclear_specific_energy(z_domain, 0.3, 0.0)


def test_compute_mcf_correction_factor_zero_limit():
    z_domain = np.zeros(4)
    z_nucleus = np.zeros(4)

    c = compute_mcf_correction_factor(
        z_domain=z_domain,
        z_nucleus=z_nucleus,
        alpha0=0.2,
        beta0=0.05,
    )

    assert np.allclose(c, 1.0)


def test_compute_mcf_correction_factor_matches_definition():
    z_domain = np.array([0.5, 1.0, 2.0])
    z_nucleus = np.array([0.05, 0.10, 0.20])
    alpha0 = 0.2
    beta0 = 0.05

    T = (
        alpha0 * z_nucleus
        + beta0 * z_domain * z_nucleus
        + beta0 * z_nucleus ** 2
    )
    expected = (1.0 - np.exp(-T)) / T

    result = compute_mcf_correction_factor(
        z_domain=z_domain,
        z_nucleus=z_nucleus,
        alpha0=alpha0,
        beta0=beta0,
    )

    assert np.allclose(result, expected)


def test_compute_mcf_correction_factor_requires_matching_shapes():
    with pytest.raises(ValueError, match="same shape"):
        compute_mcf_correction_factor(
            z_domain=np.ones(3),
            z_nucleus=np.ones(4),
            alpha0=0.2,
            beta0=0.05,
        )


def test_compute_mcf_averaged_quantities_matches_definition():
    b = np.linspace(0.0, 2.0, 101)
    z_domain = np.exp(-b)
    c = 0.8 + 0.1 * np.exp(-0.5 * b)

    denom = _trapezoid(z_domain * b, b)
    expected_c_bar = _trapezoid(c * z_domain * b, b) / denom
    expected_z_bar_c = _trapezoid(z_domain ** 2 * c * b, b) / denom

    c_bar, z_bar_c = compute_mcf_averaged_quantities(
        z_domain=z_domain,
        b_array=b,
        c=c,
        integration_method="trapz",
    )

    assert np.isclose(c_bar, expected_c_bar)
    assert np.isclose(z_bar_c, expected_z_bar_c)


def test_mcf_alpha_decomposition_identity():
    """The two stored MCF averages must exactly reconstruct the spectral alpha integral."""
    b = np.linspace(0.0, 3.0, 301)
    z_domain = 2.0 * np.exp(-b)
    z_nucleus = 0.08 * np.exp(-b)
    alpha0 = 0.18
    beta0 = 0.04

    c = compute_mcf_correction_factor(
        z_domain=z_domain,
        z_nucleus=z_nucleus,
        alpha0=alpha0,
        beta0=beta0,
    )
    c_bar, z_bar_c = compute_mcf_averaged_quantities(
        z_domain=z_domain,
        b_array=b,
        c=c,
        integration_method="trapz",
    )

    alpha_from_averages = alpha0 * c_bar + beta0 * z_bar_c

    denom = _trapezoid(z_domain * b, b)
    alpha_direct = _trapezoid(
        (alpha0 + beta0 * z_domain) * c * z_domain * b,
        b,
    ) / denom

    assert np.isclose(alpha_from_averages, alpha_direct, rtol=1e-12, atol=1e-12)


def test_compute_mcf_averaged_quantities_integration_methods_are_consistent():
    b = np.linspace(0.0, 3.0, 1001)
    z_domain = np.exp(-b)
    c = 0.7 + 0.2 * np.exp(-0.3 * b)

    trapz_result = compute_mcf_averaged_quantities(z_domain, b, c, "trapz")
    simps_result = compute_mcf_averaged_quantities(z_domain, b, c, "simps")
    quad_result = compute_mcf_averaged_quantities(z_domain, b, c, "quad")

    assert np.allclose(trapz_result, simps_result, rtol=1e-5, atol=1e-7)
    assert np.allclose(trapz_result, quad_result, rtol=1e-5, atol=1e-7)


def test_compute_mcf_averaged_quantities_zero_denominator():
    b = np.linspace(0.0, 1.0, 5)
    z_domain = np.zeros_like(b)
    c = np.ones_like(b)

    assert compute_mcf_averaged_quantities(z_domain, b, c) == (0.0, 0.0)


def test_compute_mcf_averaged_quantities_validates_inputs():
    with pytest.raises(ValueError, match="same shape"):
        compute_mcf_averaged_quantities(
            z_domain=np.ones(3),
            b_array=np.ones(4),
            c=np.ones(3),
        )

    b = np.linspace(0.0, 1.0, 5)
    with pytest.raises(ValueError, match="Unsupported integration method"):
        compute_mcf_averaged_quantities(
            z_domain=np.ones(5),
            b_array=b,
            c=np.ones(5),
            integration_method="invalid",
        )

def test_compute_mcf_lq_coefficients():
    c_bar = 0.8
    z_bar_c = 0.3
    alpha0 = 0.1
    beta0 = 0.05

    alpha_mcf, beta_mcf = compute_mcf_lq_coefficients(
        c_bar=c_bar,
        z_bar_c=z_bar_c,
        alpha0=alpha0,
        beta0=beta0,
    )

    assert alpha_mcf == pytest.approx(alpha0 * c_bar + beta0 * z_bar_c)
    assert beta_mcf == pytest.approx(beta0 * c_bar ** 2)

