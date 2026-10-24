# MMA3001 Project: Six-DOF Rocket Flight Simulator

A six-degree-of-freedom (6-DOF) flight simulator for a single-stage rocket, written in Python for the MMA3001 Numerical Methods and Machine Learning individual project.

The simulator predicts apogee, stability and landing point from the rocket's design data. It is verified against analytical solutions and validated against measured flight data.

## Project status

| Component | Status |
| --- | --- |
| RK4 integrator | Done, with tests |
| Atmosphere model | Done |
| Thrust and mass model | Done |
| Rigid-body rotation (quaternions) | Done |
| Aerodynamic forces and moments | Planned |
| Validation against flight data | Planned |

## Repository structure

```text
MMA3001-Project/
├── src/rocket6dof/       Simulator source code (Python package)
│   ├── __init__.py       Marks the folder as a package
│   └── integrators.py    Fixed-step RK4 time integration
├── tests/                Automated tests (pytest)
│   ├── test_smoke.py     Checks the test setup works
│   └── test_integrators.py
├── pytest.ini            pytest settings
├── requirements.txt      Exact package versions
├── LICENSE
└── README.md
```

## Setup

Tested with Python 3.13 on Windows.

1. Clone the repository and move into it:

```
   git clone https://github.com/YOUR-USERNAME/MMA3001-Project.git
   cd MMA3001-Project
```

2. Create and activate a virtual environment:

```
   python -m venv .venv
```

   - Windows (PowerShell): `.venv\Scripts\Activate.ps1`
   - Windows (Command Prompt): `.venv\Scripts\activate.bat`
   - macOS or Linux: `source .venv/bin/activate`

   If PowerShell says running scripts is disabled, run this once and try again:

```
   Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

3. Install the exact package versions used in this project:

```
   pip install -r requirements.txt

4. pip install e .
```

## Running the tests

With the environment active, run:

```
pytest
```

All tests should pass. The integrator tests check the RK4 method against an exact ballistic trajectory, measure its convergence order (expected: about 4), and check that invalid inputs are rejected.

## AI use

This project was developed with assistance from Claude (Anthropic). How AI was used, and how its output was checked, is described in the AI reflection section of the project report.

## License

See [LICENSE](LICENSE).