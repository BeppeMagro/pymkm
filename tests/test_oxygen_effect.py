import numpy as np
import pytest
from types import SimpleNamespace
from pymkm.biology.oxygen_effect import (
    compute_relative_radioresistance,
    _compute_scaling_factor_component,
    compute_scaling_factors,
    compute_osmk2021_radioresistance,
    compute_osmk2023_radioresistance,
    compute_osmk_radioresistance,
    compute_osmk_alpha,
    compute_osmk_beta,
    apply_oxygen_correction_alpha,
    apply_oxygen_correction_beta,
    compute_osmk_lq_coefficients,
    compute_osmk2023_effective_parameters,
)

def test_compute_relative_radioresistance_scalar():
    K = 3.0
    pO2 = 5.0
    K_mult = np.array([2.0])  # Scalar in array
    R = compute_relative_radioresistance(K, pO2, K_mult)
    expected = (5.0 + 3.0) / (5.0 + 3.0 * 2.0)
    assert np.allclose(R, expected)

def test_compute_relative_radioresistance_vector():
    K = 3.0
    pO2 = 5.0
    K_mult = np.array([1.0, 2.0, 3.0])
    R = compute_relative_radioresistance(K, pO2, K_mult)
    expected = (5 + K) / (5 + K * K_mult)
    assert np.allclose(R, expected)

def test_compute_scaling_factor_component():
    R = np.array([1.0, 1.5, 2.0])
    f_max = 1.6
    Rmax = 2.0
    scale = _compute_scaling_factor_component(R, f_max, Rmax)
    expected = 1 + (R - 1) * (f_max - 1) / (Rmax - 1)
    assert np.allclose(scale, expected)

def test_compute_scaling_factors():
    R = np.array([1.0, 1.5, 2.0])
    f_rd_max = 1.2
    f_z0_max = 2.0
    Rmax = 2.0
    f_rd, f_z0 = compute_scaling_factors(R, f_rd_max, f_z0_max, Rmax)
    expected_rd = _compute_scaling_factor_component(R, f_rd_max, Rmax)
    expected_z0 = _compute_scaling_factor_component(R, f_z0_max, Rmax) ** 2
    assert np.allclose(f_rd, expected_rd)
    assert np.allclose(f_z0, expected_z0)

def test_compute_osmk_radioresistance_2021():
    params = SimpleNamespace(
        K=3.0,
        pO2=5.0,
        zR=0.5,
        gamma=2.0,
        Rm=2.0
    )
    z_bar_domain = 1.0
    R, f_rd, f_z0 = compute_osmk_radioresistance("2021", z_bar_domain, params)

    assert isinstance(R, float)
    assert f_rd is None
    assert f_z0 is None

def test_compute_osmk_radioresistance_2023():
    params = SimpleNamespace(
        K=3.0,
        pO2=5.0,
        Rmax=2.0,
        f_rd_max=1.5,
        f_z0_max=2.0
    )
    z_bar_domain = 1.0
    R, f_rd, f_z0 = compute_osmk_radioresistance("2023", z_bar_domain, params)

    assert isinstance(R, float)
    assert isinstance(f_rd, float)
    assert isinstance(f_z0, float)

def test_compute_osmk_radioresistance_invalid_version():
    with pytest.raises(ValueError, match="Unsupported OSMK version: dummy"):
        compute_osmk_radioresistance("dummy", np.array([1.0]), SimpleNamespace())

def test_apply_oxygen_correction_alpha():
    z_bar_star = 2.0
    R = 2.0
    params = SimpleNamespace(alphaL=0.02, alphaS=0.06, beta0=0.05)

    result = apply_oxygen_correction_alpha(z_bar_star, R, params)
    expected = 0.02 + (0.06 / R) + (0.05 * z_bar_star / (R**2))
    assert abs(result - expected) < 1e-10

def test_apply_oxygen_correction_beta():
    z_bar_star = 3.0
    z_bar = 1.5
    R = 2.0
    params = SimpleNamespace(beta0=0.04)

    result = apply_oxygen_correction_beta(z_bar_star, z_bar, R, params)
    expected = (z_bar_star / z_bar) * 0.04 / (R**2)
    assert abs(result - expected) < 1e-10


def test_compute_osmk_lq_coefficients():
    z_bar_star = 3.0
    z_bar = 1.5
    R = 2.0
    params = SimpleNamespace(alphaL=0.02, alphaS=0.06, beta0=0.04)

    alpha, beta = compute_osmk_lq_coefficients(z_bar_star, z_bar, R, params)

    expected_alpha = 0.02 + (0.06 / R) + (0.04 * z_bar_star / (R ** 2))
    expected_beta = (z_bar_star / z_bar) * 0.04 / (R ** 2)
    assert alpha == pytest.approx(expected_alpha)
    assert beta == pytest.approx(expected_beta)


def test_compute_osmk_lq_coefficients_zero_domain_is_safe():
    params = SimpleNamespace(alphaL=0.02, alphaS=0.06, beta0=0.04)
    alpha, beta = compute_osmk_lq_coefficients(0.0, 0.0, 2.0, params)
    assert alpha == pytest.approx(0.05)
    assert float(beta) == pytest.approx(0.0)

def test_compute_osmk2023_effective_parameters_uses_shared_precision():
    rd_eff, z0_eff = compute_osmk2023_effective_parameters(
        domain_radius=0.3,
        z0=0.85,
        f_rd=1.23456,
        f_z0=1.54321,
    )

    assert rd_eff == round(0.3 / 1.23456, 3)
    assert z0_eff == round(0.85 * 1.54321, 2)

def test_compute_osmk2021_radioresistance_explicit_matches_compatibility_wrapper():
    params = SimpleNamespace(
        K=3.0,
        pO2=5.0,
        zR=0.5,
        gamma=2.0,
        Rm=2.0,
    )
    z_bar_domain = 1.0

    direct = compute_osmk2021_radioresistance(
        z_bar_domain=z_bar_domain,
        K=params.K,
        pO2=params.pO2,
        zR=params.zR,
        gamma=params.gamma,
        Rm=params.Rm,
    )
    wrapped, f_rd, f_z0 = compute_osmk_radioresistance(
        "2021", z_bar_domain, params
    )

    assert direct == pytest.approx(wrapped)
    assert f_rd is None
    assert f_z0 is None


def test_compute_osmk2023_radioresistance_explicit_matches_compatibility_wrapper():
    params = SimpleNamespace(
        K=3.0,
        pO2=5.0,
        Rmax=2.0,
        f_rd_max=1.5,
        f_z0_max=2.0,
    )

    direct = compute_osmk2023_radioresistance(
        K=params.K,
        pO2=params.pO2,
        Rmax=params.Rmax,
        f_rd_max=params.f_rd_max,
        f_z0_max=params.f_z0_max,
    )
    wrapped = compute_osmk_radioresistance("2023", 1.0, params)

    assert np.allclose(direct, wrapped)


def test_compute_osmk_alpha_and_beta_explicit():
    z_bar_star = 3.0
    z_bar = 1.5
    R = 2.0

    alpha = compute_osmk_alpha(
        z_bar_star_domain=z_bar_star,
        R=R,
        alphaL=0.02,
        alphaS=0.06,
        beta0=0.04,
    )
    beta = compute_osmk_beta(
        z_bar_star_domain=z_bar_star,
        z_bar_domain=z_bar,
        R=R,
        beta0=0.04,
    )

    assert alpha == pytest.approx(0.02 + 0.06 / R + 0.04 * z_bar_star / R**2)
    assert beta == pytest.approx((z_bar_star / z_bar) * 0.04 / R**2)


def test_compute_osmk_lq_coefficients_explicit_parameters():
    alpha, beta = compute_osmk_lq_coefficients(
        z_bar_star_domain=3.0,
        z_bar_domain=1.5,
        R=2.0,
        alphaL=0.02,
        alphaS=0.06,
        beta0=0.04,
    )

    assert alpha == pytest.approx(0.02 + 0.06 / 2.0 + 0.04 * 3.0 / 2.0**2)
    assert beta == pytest.approx((3.0 / 1.5) * 0.04 / 2.0**2)


def test_compute_osmk_lq_coefficients_rejects_mixed_parameter_styles():
    params = SimpleNamespace(alphaL=0.02, alphaS=0.06, beta0=0.04)

    with pytest.raises(ValueError, match="either params or explicit"):
        compute_osmk_lq_coefficients(
            3.0,
            1.5,
            2.0,
            params,
            alphaL=0.02,
            alphaS=0.06,
            beta0=0.04,
        )


def test_compute_osmk_lq_coefficients_requires_complete_explicit_parameters():
    with pytest.raises(ValueError, match="must all be provided"):
        compute_osmk_lq_coefficients(
            3.0,
            1.5,
            2.0,
            alphaL=0.02,
            alphaS=0.06,
        )

