import numpy as np
import pytest

from rocket6dof.wind import Wind


def test_west_wind_blows_towards_east():
    """A 270 deg wind comes from the west, so it blows towards +x (east)."""
    np.testing.assert_allclose(Wind(5.0, 270.0)(100.0), [5.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(Wind(5.0, 0.0)(100.0), [0.0, -5.0, 0.0], atol=1e-12)


def test_power_law_profile():
    w = Wind(4.0, 270.0, exponent=1.0 / 7.0, reference_height=10.0)
    assert w(10.0)[0] == pytest.approx(4.0)
    assert w(1280.0)[0] == pytest.approx(4.0 * 128.0 ** (1.0 / 7.0))


@pytest.mark.parametrize("kwargs", [{"speed": -1.0}, {"speed": 1.0, "exponent": -0.1},
                                    {"speed": 1.0, "reference_height": 0.0}])
def test_rejects_invalid_wind(kwargs):
    with pytest.raises(ValueError):
        Wind(**kwargs)
