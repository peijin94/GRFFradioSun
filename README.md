# GRFFradioSun

Synthetic radio free-free emission from a MAS coronal model using GRFF.

## Requirements

- **Python**: `psipy`, `numpy`, `matplotlib`, `astropy`, `xarray`
- **Data**: MAS model in `corona/` (e.g. `rho002.hdf`, `t002.hdf`, `br002.hdf`, etc.)
- **GRFF**: `GRFF/binaries/GRFF_DEM_Transfer.so`
- **Optional GPU**: NVIDIA CUDA + CuPy (example: `pip install cupy-cuda12x`)

## Layout

- `raytracingGRFF/`: installable Python package (ray tracing and LOS sampling code)
- `script/`: runnable workflows
- `fastGRFF/`: external module; not included in `raytracingGRFF` packaging

## Environment

```bash
source /home/pjzhang/miniconda3/etc/profile.d/conda.sh
conda activate lwa
```

## Scripts

### `script/resampling_MAS_LOS.py`

Non-raytracing baseline: resample MAS along straight LOS and write `LOS_data.npz`.

Example:

```bash
python script/resampling_MAS_LOS.py -m ./corona -o LOS_data_75MHz.npz --dz0 7e-4 -f 2.2
```

Parameters:

| Flag | Type | Default | Description |
|---|---|---|---|
| `-m`, `--model-path` | `str` | `./corona` | Path to MAS model directory. |
| `-n`, `--N-pix` | `int` | `256` | Image size `N_pix x N_pix`. |
| `-f`, `--X-FOV` | `float` | `1.44` | Half field-of-view in `R_sun`; x,y in `[-X-FOV, X-FOV]`. |
| `-z`, `--N-z` | `int` | `400` | Number of samples along each LOS. |
| `-d`, `--dz0` | `float` | `3e-4` | Initial spacing for irregular z-grid, in `R_sun`. |
| `-v`, `--no-variable-spacing-z` | flag | `False` | Use regular linear z spacing instead of irregular spacing. |
| `-zr`, `--z-range` | `min,max` | `None` | Z extent in `R_sun` for linear spacing mode. |
| `-o`, `--out-path` | `str` | `LOS_data.npz` | Output LOS `.npz` path. |
| `-p`, `--no-plots` | flag | `False` | Disable LOS profile/slice plots. |
| `-q`, `--quiet` | flag | `False` | Suppress progress messages. |

Note: `--dz0` is in `R_sun`; `7e4` is invalid for this use case. Use values like `7e-4`.

### `script/synthetic_FF_map_single_thread.py`

Non-raytracing baseline: compute synthetic free-free map from LOS `.npz` using GRFF.

Example:

```bash
python script/synthetic_FF_map_single_thread.py -i LOS_data_75MHz.npz -o emission_map
```

Parameters:

| Flag | Type | Default | Description |
|---|---|---|---|
| `-i`, `--input` | `str` | `LOS_data.npz` | Input LOS `.npz` file. |
| `-o`, `--output` | `str` | `emission_map` | Output base path (no extension). |
| `-f`, `--freq0` | `float` | `450e6` | Start frequency in Hz. |
| `-n`, `--Nfreq` | `int` | `4` | Number of frequency channels. |
| `-s`, `--freq-log-step` | `float` | `0.1` | `log10` step between frequencies. |
| `--do-inspection-plot` | flag | `False` | Save center-pixel LOS inspection plot. |

### `script/resample_with_ray_tracing.py`

Raytracing workflow: resample MAS onto cube, trace rays, sample LOS, then run GRFF.

Example (CPU):

```bash
python script/resample_with_ray_tracing.py -m ./corona -o ray_tracing_emission.npz --device cpu --raytrace-device cpu
```

Example (GPU + external `fastGRFF` backend):

```bash
python script/resample_with_ray_tracing.py --device cuda --raytrace-device cuda --grff-backend fastgrff
```

Parameters:

| Flag | Type | Default | Description |
|---|---|---|---|
| `-m`, `--model-path` | `str` | `./corona` | MAS model directory. |
| `-n`, `--N-pix` | `int` | `64` | Image size `N_pix x N_pix`. |
| `-f`, `--X-FOV` | `float` | `1.44` | Half FOV in `R_sun`. |
| `--freq` | `float` | `75e6` | Ray frequency in Hz. |
| `--grid-n` | `int` | `128` | 3D cube points per axis. |
| `--grid-extent` | `float` | `3.0` | Cube extent in `R_sun` (`[-extent, extent]`). |
| `--z-observer` | `float` | `3.0` | Ray start z in `R_sun`. |
| `--dt` | `float` | `6e-3` | Ray integration timestep. |
| `--n-steps` | `int` | `5000` | Number of ray integration steps. |
| `--record-stride` | `int` | `10` | Record every N integration steps. |
| `-w`, `--workers` | `int` | `1` | Processes for parallel ray tracing. |
| `-o`, `--out-path` | `str` | `ray_tracing_emission.npz` | Output `.npz` path. |
| `--grff-lib` | `str` | `GRFF/binaries/GRFF_DEM_Transfer.so` | Path to GRFF shared library. |
| `--grff-backend` | `str` | `get_mw` | `get_mw` or `fastgrff`. |
| `--s-input-on` | flag | `False` | Pass cross-section ratio `S` into `Parms[14]`. |
| `--device` | `str` | `cpu` | LOS sampler device: `cpu` or `cuda`. |
| `--raytrace-device` | `str` | `cpu` | Ray integrator device: `cpu` or `cuda`. |
| `--no-fallback` | flag | `False` | Disable CUDA-to-CPU fallback. |
| `--no-plots` | flag | `False` | Disable emission map plot output. |
| `-q`, `--quiet` | flag | `False` | Reduce log output. |

## Validation

```bash
python -m pytest -q tests/test_gpu_raytrace.py
python bench_raytrace.py --n-pix 256 --n-steps 256
```
