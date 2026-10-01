import numpy as np
import pytest

from pymkm.utils.geometry_tools import DEFAULT_BASE_POINTS, GeometryTools


# --- Tests for calculate_squared_radius_ratio ---

def test_calculate_squared_radius_ratio():
    result = GeometryTools.calculate_squared_radius_ratio(0.5, 5.0)
    assert np.isclose(result, 0.01)

def test_calculate_squared_radius_ratio_requires_positive_radii():
    with pytest.raises(ValueError, match="Both radii must be positive"):
        GeometryTools.calculate_squared_radius_ratio(0.0, 5.0)

    with pytest.raises(ValueError, match="Both radii must be positive"):
        GeometryTools.calculate_squared_radius_ratio(0.5, 0.0)

def test_default_base_points_constant():
    assert DEFAULT_BASE_POINTS == 150


# --- Tests for determine_sampling_points ---

def test_determine_sampling_points_multiplier_20():
    # Trigger condition: energy <= 10 OR radius_max <= 0.05.
    # Here, energy=15 (>10) but radius_max=0.04 (<=0.05) should trigger multiplier=20.
    base = DEFAULT_BASE_POINTS
    result = GeometryTools.determine_sampling_points(15, 0.04)
    expected = base * 20
    assert result == expected

def test_determine_sampling_points_multiplier_15():
    # Trigger condition: energy <= 20 OR radius_max <= 0.1.
    # Use energy=25 (>20) and radius_max=0.09 (<=0.1) to trigger multiplier=15.
    base = DEFAULT_BASE_POINTS
    result = GeometryTools.determine_sampling_points(25, 0.09)
    expected = base * 15
    assert result == expected

def test_determine_sampling_points_multiplier_12():
    # Trigger condition: energy <= 40 OR radius_max <= 0.5.
    # Use energy=35 (<=40) and radius_max=1.0 (>0.5) to trigger multiplier=12.
    base = DEFAULT_BASE_POINTS
    result = GeometryTools.determine_sampling_points(35, 1.0)
    expected = base * 12
    assert result == expected

def test_determine_sampling_points_multiplier_8():
    # Trigger condition: energy <= 60 OR radius_max <= 1.
    # Use energy=50 (>40 and <=60) and radius_max=10 to trigger multiplier=8.
    base = DEFAULT_BASE_POINTS
    result = GeometryTools.determine_sampling_points(50, 10)
    expected = base * 8
    assert result == expected

def test_determine_sampling_points_multiplier_6():
    # Trigger condition: energy <= 80 OR radius_max <= 2.
    # Use energy=90 (>80) and radius_max=1.5 (<=2) to trigger multiplier=6.
    base = DEFAULT_BASE_POINTS
    result = GeometryTools.determine_sampling_points(90, 1.5)
    expected = base * 6
    assert result == expected

def test_determine_sampling_points_multiplier_5():
    # Trigger condition: energy <= 100 OR radius_max <= 5.
    # Use energy=110 (>100) and radius_max=4 (<=5) to trigger multiplier=5.
    base = DEFAULT_BASE_POINTS
    result = GeometryTools.determine_sampling_points(110, 4)
    expected = base * 5
    assert result == expected

def test_determine_sampling_points_multiplier_4():
    # Trigger condition: energy <= 150 OR radius_max <= 10.
    # Use energy=145 (<=150) and radius_max=15 (>10) to trigger multiplier=4.
    base = DEFAULT_BASE_POINTS
    result = GeometryTools.determine_sampling_points(145, 15)
    expected = base * 4
    assert result == expected

def test_determine_sampling_points_multiplier_3():
    # Trigger condition: energy <= 200 OR radius_max <= 20.
    # Use energy=210 (>200) but radius_max=18 (<=20) to trigger multiplier=3.
    base = DEFAULT_BASE_POINTS
    result = GeometryTools.determine_sampling_points(210, 18)
    expected = base * 3
    assert result == expected

def test_determine_sampling_points_multiplier_2():
    # Else branch: energy > 200 AND radius_max > 20.
    # Use energy=210 and radius_max=25 to trigger multiplier=2.
    base = DEFAULT_BASE_POINTS
    result = GeometryTools.determine_sampling_points(210, 25)
    expected = base * 2
    assert result == expected

def test_determine_sampling_points_custom_base():
    # Test using a custom base_points value.
    custom_base = 200
    # For energy=5, radius_max=0.03, condition holds for multiplier=20.
    result = GeometryTools.determine_sampling_points(5, 0.03, base_points=custom_base)
    expected = custom_base * 20
    assert result == expected

# --- Tests for generate_default_radii ---

def test_generate_default_radii_length_default():
    energy = 50.0
    radius_max = 10.0
    radii = GeometryTools.generate_default_radii(energy, radius_max)
    N = GeometryTools.determine_sampling_points(energy, radius_max)
    assert radii.shape[0] == N

def test_generate_default_radii_custom_base():
    energy = 50.0
    radius_max = 10.0
    custom_base = 200
    radii = GeometryTools.generate_default_radii(energy, radius_max, base_points=custom_base)
    N = GeometryTools.determine_sampling_points(energy, radius_max, base_points=custom_base)
    assert radii.shape[0] == N

def test_generate_default_radii_raises_value_error():
    with pytest.raises(ValueError, match="Both energy and maximum radius must be provided."):
        GeometryTools.generate_default_radii(None, 10.0)
    with pytest.raises(ValueError, match="Both energy and maximum radius must be provided."):
        GeometryTools.generate_default_radii(50.0, None)

# --- Tests for calculate_intersection_area ---

def test_calculate_intersection_area_no_overlap():
    r1 = np.array([1, 2, 3])
    r2 = 1.0
    d = 10.0  # No overlap
    area = GeometryTools.calculate_intersection_area(r1, r2, d)
    np.testing.assert_allclose(area, np.zeros_like(r1))

def test_calculate_intersection_area_contained():
    r1 = np.array([5.0])
    r2 = 2.0
    d = 1.0  # d <= |5-2|, so the smaller circle is completely inside
    area = GeometryTools.calculate_intersection_area(r1, r2, d)
    expected = np.pi * (min(5.0, 2.0) ** 2)
    np.testing.assert_allclose(area, np.array([expected]), rtol=1e-5)

def test_calculate_intersection_area_overlap():
    # Two circles with equal radii (r = 2) and center distance d = 2.
    # Analytical intersection area: A = 8*(pi/3) - 2*sqrt(3)
    r1 = np.array([2.0])
    r2 = 2.0
    d = 2.0
    area = GeometryTools.calculate_intersection_area(r1, r2, d)
    expected = 8 * (np.pi / 3) - 2 * np.sqrt(3)
    np.testing.assert_allclose(area, np.array([expected]), rtol=1e-5)
