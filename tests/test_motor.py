import pytest

from rocket6dof.motor import Motor, load_eng

#A made up triangular thrust curve: thrust rises form 0 N at t =0s to 100 N at t =1s, 
# then falls back to 0N at t=2s. The point (0,0) is added by Motor. Total impulse 
# is the triangle's area, 100 N s.
TIMES = [1.0,2.0]
THRUSTS = [100.0,0.0]
PROPELLANT_MASS = 0.05
TOTAL_MASS = 0.15

def make_motor():
    return Motor("TEST", TIMES, THRUSTS, PROPELLANT_MASS, TOTAL_MASS)

def test_thrust_is_linear_between_points():
    motor = make_motor()
    assert motor.thrust(0.5) == pytest.approx(50.0)
    assert motor.thrust(1.0) == pytest.approx(100.0)
    assert motor.thrust(1.5) == pytest.approx(50.0)

def test_thrust_is_zero_outside_burn():
    motor=make_motor()
    assert motor.thrust(-1.0) == 0
    assert motor.thrust(3.0) == 0

def test_total_impulse_is_triangle_area():
    assert make_motor().total_impulse == pytest.approx(100.0)

def test_mass_follows_impulse_delivered():
    motor=make_motor()
    # By t = 0.5 s the impulse is 0.5 * 0.5 s * 50 N = 12.5 N s, which is
    # 12.5% of the total, so 12.5% of the propellant has burned.
    assert motor.mass(0.0) == pytest.approx(TOTAL_MASS)
    assert motor.mass(0.5) == pytest.approx(TOTAL_MASS - 0.125 * PROPELLANT_MASS)
    assert motor.mass(1.0) == pytest.approx(TOTAL_MASS - 0.5 * PROPELLANT_MASS)
    assert motor.mass(2.0) == pytest.approx(TOTAL_MASS - PROPELLANT_MASS)
    assert motor.mass(10.0) == pytest.approx(TOTAL_MASS - PROPELLANT_MASS)

def test_load_eng_reads_file(tmp_path):
    eng_text = (
        "; made up motor for testing\n"
        "test 29 100 0 0.05 0.15 testco\n"
        "1.0 100.0\n"
        "2.0 0.0\n"
    )
    path=tmp_path/"test.eng"
    path.write_text(eng_text)

    motor = load_eng(path)
    assert motor.name == "test"
    assert motor.propellant_mass == pytest.approx(0.05)
    assert motor.total_mass == pytest.approx(0.15)
    assert motor.total_impulse== pytest.approx(100.0)

@pytest.mark.parametrize(
    "times,thrusts",
    [
        ([1.0, 0.5, 2.0], [50.0, 100.0, 0.0]),  # times not increasing
        ([1.0, 2.0], [-10.0, 0.0]),             # negative thrust
        ([1.0, 2.0], [100.0, 20.0]),            # does not end at zero
    ],
)
def test_rejects_invalid_thrust_curve(times,thrusts):
    with pytest.raises(ValueError):
        Motor("BAD", times, thrusts, PROPELLANT_MASS, TOTAL_MASS)

def tests_rejects_propellant_heavier_than_motor():
    with pytest.raises(ValueError):
        Motor("BAD", TIMES, THRUSTS, 0.2, 0.15)