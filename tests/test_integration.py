import numpy as np
import pytest

from pymkm.utils.integration import integrate_1d


def test_integrate_1d_trapz_prefers_numpy_trapezoid(monkeypatch):
    x = np.linspace(0.0, 1.0, 11)
    y = x ** 2

    def fake_trapezoid(y_array, x_array):
        assert y_array is y
        assert x_array is x
        return 0.123

    monkeypatch.setattr(np, "trapezoid", fake_trapezoid, raising=False)
    assert integrate_1d(y, x, method="trapz") == pytest.approx(0.123)


def test_integrate_1d_trapz_falls_back_for_older_numpy(monkeypatch):
    x = np.linspace(0.0, 1.0, 11)
    y = x ** 2

    def fake_trapz(y_array, x_array):
        assert y_array is y
        assert x_array is x
        return 0.456

    monkeypatch.delattr(np, "trapezoid", raising=False)
    monkeypatch.setattr(np, "trapz", fake_trapz)
    assert integrate_1d(y, x, method="trapz") == pytest.approx(0.456)

def test_integrate_1d_simps():
    x = np.linspace(0.0, 1.0, 101)
    y = x ** 2
    result = integrate_1d(y, x, method="simps")
    assert result == pytest.approx(1.0 / 3.0, rel=1e-8)


def test_integrate_1d_quad():
    x = np.linspace(0.0, 1.0, 101)
    y = x ** 2
    result = integrate_1d(y, x, method="quad")
    assert result == pytest.approx(1.0 / 3.0, rel=1e-8)


def test_integrate_1d_invalid_method():
    x = np.linspace(0.0, 1.0, 5)
    y = np.ones_like(x)
    with pytest.raises(ValueError, match="Unsupported integration method"):
        integrate_1d(y, x, method="invalid")
