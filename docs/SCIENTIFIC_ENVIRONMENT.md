# Scientific runtime environment

This repository requires a dedicated Python 3.11 environment. The system-installed Python 3.14 environment is not a valid scientific runtime for the SatQuery AI pipeline because NumPy and PyTorch fail there.

## Active project environment

The repo-local environment is kept at:

- `.venv-scientific`

Use it with:

```powershell
cd E:\SatQuery_ai_2.0\Satquery_ai_2.0
.\.venv-scientific\Scripts\Activate.ps1
```

## Recreate the environment

```powershell
py -3.11 -m venv .venv-scientific
.\.venv-scientific\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\.venv-scientific\Scripts\python.exe -m pip install -r requirements-scientific.txt
```

## Verified stack

The environment was validated with Python 3.11.15 and the following packages:

- numpy 2.4.6
- torch 2.14.0+cpu
- pandas 3.0.5
- pyarrow 25.0.1
- rasterio 1.4.4
- matplotlib 3.11.2
- scikit-learn (current compatible release in the venv)

This is a runtime repair only; no scientific model logic, dataset logic, or pipeline architecture was changed.