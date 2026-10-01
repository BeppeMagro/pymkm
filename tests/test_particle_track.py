import numpy as np
import pytest
import logging

from pymkm.physics.particle_track import ParticleTrack
from pymkm.utils.geometry_tools import DEFAULT_BASE_POINTS, GeometryTools

# --- Helper Functions ---

def expected_velocity(energy: float) -> float:
    m0 = 931.5
    return np.sqrt(1 - 1 / ((1 + energy / m0) ** 2))

def test_default_base_points_uses_shared_constant():
    pt = ParticleTrack(
        model_name="Kiefer-Chatterjee",
        energy=150,
        atomic_number=6,
        let=50,
    )
    assert pt.base_points == DEFAULT_BASE_POINTS


# --- Tests for Initialization and Warning Branches ---

def test_invalid_core_radius_type():
    with pytest.raises(ValueError, match=r"Invalid core_radius_type\."):
        ParticleTrack(model_name='Kiefer-Chatterjee', core_radius_type='invalid',
                      energy=150, atomic_number=6, let=50)

def test_invalid_model_name():
    with pytest.raises(ValueError, match=r"Invalid model_name\."):
        ParticleTrack(model_name='InvalidModel', core_radius_type='energy-dependent',
                      energy=150, atomic_number=6, let=50)

def test_warning_kiefer_constant(caplog):
    caplog.set_level(logging.WARNING)
    pt = ParticleTrack(model_name='Kiefer-Chatterjee', core_radius_type='constant',
                       energy=150, atomic_number=6, let=50)
    # Check that a warning was logged about support only for energy-dependent
    assert "supports only 'energy-dependent' core radius" in caplog.text
    # Ensure core_radius_type was switched to 'energy-dependent'
    assert pt.core_radius_type == 'energy-dependent'

def test_warning_scholz_energy_dependent_switch(caplog):
    caplog.set_level(logging.WARNING)
    pt = ParticleTrack(model_name='Scholz-Kraft', core_radius_type='energy-dependent',
                       energy=150, let=50)
    assert "supports only 'constant' core radius" in caplog.text
    assert pt.core_radius_type == 'constant'

def test_missing_energy_raises_in_init_scholz():
    with pytest.raises(ValueError, match=r"Kinetic energy is required to compute velocity\."):
        ParticleTrack(model_name='Scholz-Kraft', core_radius_type='constant',
                      energy=None, let=50)

def test_missing_atomic_number_kiefer():
    with pytest.raises(ValueError, match=r"Atomic number must be provided"):
        ParticleTrack(model_name='Kiefer-Chatterjee', core_radius_type='energy-dependent',
                      energy=150, atomic_number=None, let=50)

# --- Tests for Internal Computations ---

def test_convert_let():
    pt = ParticleTrack(model_name='Kiefer-Chatterjee', core_radius_type='energy-dependent',
                       energy=150, atomic_number=6, let=100)
    conv = pt._convert_let(100)
    expected = 100 * 1e-7 * 1.602176462e-7
    np.testing.assert_allclose(conv, expected, rtol=1e-5)

def test_missing_energy_for_velocity_in_init():
    with pytest.raises(ValueError, match=r"Kinetic energy is required to compute velocity\."):
        ParticleTrack(model_name='Kiefer-Chatterjee', core_radius_type='energy-dependent',
                      energy=None, atomic_number=6, let=50)

def test_missing_energy_for_velocity_direct():
    # Create a valid instance first.
    pt = ParticleTrack(model_name='Kiefer-Chatterjee', core_radius_type='energy-dependent',
                       energy=150, atomic_number=6, let=50)
    # Then override energy to None to force the error in _compute_velocity.
    pt.energy = None
    with pytest.raises(ValueError, match=r"Kinetic energy is required to compute velocity\."):
        pt._compute_velocity()

def test_compute_velocity():
    energy = 150
    pt = ParticleTrack(model_name='Kiefer-Chatterjee', core_radius_type='energy-dependent',
                       energy=energy, atomic_number=6, let=50)
    vel = pt._compute_velocity()
    exp_vel = expected_velocity(energy)
    np.testing.assert_allclose(vel, exp_vel, rtol=1e-3)

def test_compute_core_radius_constant_scholz():
    pt = ParticleTrack(model_name='Scholz-Kraft', core_radius_type='constant',
                       energy=150, let=50)
    cr = pt._compute_core_radius()
    np.testing.assert_allclose(cr, 1.0e-2)

def test_compute_core_radius_constant_elsaesser_scholz():
    pt = ParticleTrack(model_name='Elsaesser–Scholz', core_radius_type='constant',
                       energy=150, let=50)
    cr = pt._compute_core_radius()
    np.testing.assert_allclose(cr, 3.0e-4)

def test_compute_core_radius_constant_friedrich():
    pt = ParticleTrack(model_name='Friedrich', core_radius_type='constant',
                       energy=150, let=50)
    cr = pt._compute_core_radius()
    np.testing.assert_allclose(cr, 6.5e-2)

def test_kiefer_dose_requires_energy_dependent_core_radius():
    # Create a valid instance first.
    pt = ParticleTrack(model_name='Kiefer-Chatterjee', core_radius_type='energy-dependent',
                       energy=150, atomic_number=6, let=50)
    # Force core_radius_type to an invalid value for the Kiefer-Chatterjee model.
    pt.core_radius_type = 'constant'
    dummy_radii = np.linspace(0.1, 5, 100).reshape(-1, 1)
    with pytest.raises(ValueError, match=r"Kiefer-Chatterjee model requires energy-dependent core radius\."):
        pt._kiefer_chatterjee_dose(dummy_radii)

def test_compute_core_radius_energy_dependent_no_floor():
    energy = 150
    pt = ParticleTrack(model_name='Kiefer-Chatterjee', core_radius_type='energy-dependent',
                       energy=energy, atomic_number=6, let=50)
    Rc0 = 11.6e-3
    expected = Rc0 * pt._compute_velocity()
    computed = pt._compute_core_radius()
    np.testing.assert_allclose(computed, expected, rtol=1e-3)

def test_compute_core_radius_invalid():
    pt = ParticleTrack(model_name='Kiefer-Chatterjee', core_radius_type='energy-dependent',
                       energy=150, atomic_number=6, let=50)
    pt.core_radius_type = "invalid"
    with pytest.raises(ValueError, match=r"Unsupported core radius type\."):
        pt._compute_core_radius()

def test_invalid_model_name_penumbra_radius():
    pt = ParticleTrack(model_name='Kiefer-Chatterjee', core_radius_type='energy-dependent',
                       energy=150, atomic_number=6, let=50)
    pt.model_name = 'InvalidModel'
    with pytest.raises(ValueError, match=r"Invalid model name for penumbra radius computation\."):
        pt._compute_penumbra_radius()

def test_compute_penumbra_radius_positive():
    energy = 150
    pt = ParticleTrack(model_name='Kiefer-Chatterjee', core_radius_type='energy-dependent',
                       energy=energy, atomic_number=6, let=50)
    pr = pt._compute_penumbra_radius()
    assert pr > 0

def test_compute_effective_atomic_number():
    pt = ParticleTrack(model_name='Kiefer-Chatterjee', core_radius_type='energy-dependent',
                       energy=150, atomic_number=6, let=50)
    zeff = pt._compute_effective_atomic_number()
    assert zeff > 0

def test_missing_atomic_number_for_effective_atomic_number():
    pt = ParticleTrack(model_name='Kiefer-Chatterjee', core_radius_type='energy-dependent',
                       energy=150, atomic_number=6, let=50)
    pt.atomic_number = None
    with pytest.raises(ValueError, match=r"Atomic number is required to compute effective atomic number\."):
        pt._compute_effective_atomic_number()

def test_compute_Kp():
    pt = ParticleTrack(model_name='Kiefer-Chatterjee', core_radius_type='energy-dependent',
                       energy=150, atomic_number=6, let=50)
    kp = pt._compute_Kp()
    assert kp > 0

def test_compute_lambda0():
    pt = ParticleTrack(model_name='Kiefer-Chatterjee', core_radius_type='energy-dependent',
                       energy=150, atomic_number=6, let=50)
    lam0 = pt._compute_lambda0()
    assert lam0 > 0

# --- Tests for initial_local_dose and Custom Base Points ---

def test_missing_energy_for_default_radii():
    pt = ParticleTrack(model_name='Kiefer-Chatterjee', core_radius_type='energy-dependent',
                       energy=150, atomic_number=6, let=50)
    pt.energy = None
    with pytest.raises(ValueError, match=r"Energy must be provided to generate default radii\."):
        pt.initial_local_dose()

def test_missing_let_for_dose():
    pt = ParticleTrack(model_name='Scholz-Kraft', core_radius_type='constant',
                       energy=150, let=None)
    with pytest.raises(ValueError, match=r"LET must be provided to compute the local dose profile\."):
        pt.initial_local_dose(radius=np.array([0.1, 0.2, 0.3]))

def test_initial_local_dose_default_kiefer():
    energy = 150
    base_points = 150
    pt = ParticleTrack(model_name='Kiefer-Chatterjee', core_radius_type='energy-dependent',
                       energy=energy, atomic_number=6, let=50, base_points=base_points)
    dose, radii = pt.initial_local_dose()
    assert radii.ndim == 2 and radii.shape[1] == 1
    np.testing.assert_allclose(dose.shape, radii.shape)
    expected_N = GeometryTools.determine_sampling_points(energy, pt.penumbra_radius, base_points=base_points)
    assert radii.shape[0] == expected_N

def test_initial_local_dose_custom_radius_kiefer():
    energy = 150
    custom_radii = np.linspace(0.1, 5, 50)
    pt = ParticleTrack(model_name='Kiefer-Chatterjee', core_radius_type='energy-dependent',
                       energy=energy, atomic_number=6, let=50)
    dose, radii = pt.initial_local_dose(radius=custom_radii)
    np.testing.assert_allclose(radii.flatten(), custom_radii)
    assert dose.shape == (50, 1)

def test_initial_local_dose_scholz_constant():
    energy = 150
    pt = ParticleTrack(model_name='Scholz-Kraft', core_radius_type='constant',
                       energy=energy, let=50)
    dose, radii = pt.initial_local_dose()
    assert dose.shape == radii.shape

def test_initial_local_dose_elsaesser_scholz():
    pt = ParticleTrack(model_name='Elsaesser–Scholz', core_radius_type='constant',
                       energy=150, let=50)
    dose, radii = pt.initial_local_dose()
    assert dose.shape == radii.shape

def test_initial_local_dose_friedrich():
    pt = ParticleTrack(model_name='Friedrich', core_radius_type='constant',
                       energy=150, let=50)
    dose, radii = pt.initial_local_dose()
    assert dose.shape == radii.shape

def test_negative_radii_error():
    energy = 150
    pt = ParticleTrack(model_name='Kiefer-Chatterjee', core_radius_type='energy-dependent',
                       energy=energy, atomic_number=6, let=50)
    with pytest.raises(ValueError, match=r"Radius values must be positive\."):
        pt.initial_local_dose(radius=np.array([-1, 2, 3]))

def test_unknown_model_name_error():
    pt = ParticleTrack(model_name='Kiefer-Chatterjee', core_radius_type='energy-dependent',
                       energy=150, atomic_number=6, let=50)
    pt.model_name = 'UnknownModel'
    with pytest.raises(ValueError, match=r"Unknown model name: UnknownModel"):
        pt.initial_local_dose()

def test_core_radius_fallback_else_returns_1e_2():
    pt = ParticleTrack(model_name='Kiefer-Chatterjee',
                       core_radius_type='energy-dependent',
                       energy=150, atomic_number=6, let=50)
    # Forzo stato incoerente per colpire il fallback:
    pt.core_radius_type = 'constant'
    pt.model_name = 'Kiefer-Chatterjee'
    cr = pt._compute_core_radius()
    assert cr == 1.0e-2

def test_penumbra_requires_energy_raises():
    pt = ParticleTrack(model_name='Scholz-Kraft',
                       core_radius_type='constant',
                       energy=150, let=50)
    pt.energy = None
    with pytest.raises(ValueError, match=r"Kinetic energy is required to compute penumbra radius\."):
        pt._compute_penumbra_radius()

def test_scholz_kernel_requires_let():
    pt = ParticleTrack(model_name='Scholz-Kraft', core_radius_type='constant',
                       energy=150, let=None)
    radii = np.array([[0.5]])  # un raggio > 0
    with pytest.raises(ValueError, match=r"LET must be provided to compute the local dose profile\."):
        pt._scholz_kraft_dose(radii)

def test_kiefer_kernel_requires_let():
    pt = ParticleTrack(model_name='Kiefer-Chatterjee', core_radius_type='energy-dependent',
                       energy=150, atomic_number=6, let=None)
    radii = np.array([[0.5]])
    with pytest.raises(ValueError, match=r"LET must be provided to compute the local dose profile\."):
        pt._kiefer_chatterjee_dose(radii)