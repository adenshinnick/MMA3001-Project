import numpy as np
import pytest

from rocket6dof.wind import Wind


def test_west_wind_blows_towards_east():
    """A 270 deg wind comes from the west, so it blows towards +x (east)."""
    np.testing.assert_allclose(Wind(5.0, 270.0)(100.0), [5.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(Wind(5.0, 0.0)(100.0), [0.0, -5.0, 0.0], atol=1e-12)


def test_rejects_negative_speed():
    with pytest.raises(ValueError):
        Wind(-1.0)
