# 2-D Hard-Sphere Particle Simulation (Python)

This script simulates N circular particles undergoing free flight and instantaneous pairwise collisions in a unit square with reflecting walls. Collisions are resolved with a coefficient of restitution (e). Particle masses may differ. The animation displays instantaneous and running-average kinetic energy.

## Features
- 2-D hard-sphere dynamics with overlap correction and inelastic/elastic impacts.
- Reflecting walls.
- Heterogeneous masses.
- Live animation; optional MP4 export.

## Installation

```bash
# Option A: venv
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Option B: conda
conda create -n particles python=3.11 -y
conda activate particles
pip install -r requirements.txt
```

> MP4 export requires FFmpeg (external):
- macOS: `brew install ffmpeg`
- Ubuntu/Debian: `sudo apt-get install ffmpeg`
- Windows (winget): `winget install Gyan.FFmpeg` (or use ffmpeg.org builds)

## Usage

```bash
# run with interactive animation
python particles.py

# run and save MP4
python particles.py --video
```

The video is saved to `particles.mp4` (30 fps, dpi=200).

## Parameters (edit in source)

| Name | Meaning | Default |
|---|---|---|
| `N` | number of particles | `50` |
| `RADIUS` | particle radius (uniform) | `0.003` |
| `DT` | time step | `0.005` |
| `STEPS` | total animation frames | `1000` |
| `SEED` | RNG seed (`None` for nondeterministic) | `2` |
| `SPEED_INIT` | initial speed scale | `0.5` |
| `BOX_MIN, BOX_MAX` | square bounds | `0.0, 1.0` |
| `ELASTIC_E` | restitution e∈(0,1] | `0.8` |

Other implementation choices:
- Masses sampled i.i.d. ~ U(0.5, 3.0).
- Two physics substeps per animation frame (set in `update()`).

## Model notes

- **Wall collisions:** velocity reflection with clamping.
- **Pairwise collisions:** impulse resolved along normal with coefficient of restitution.
- **Overlap correction:** displacement proportional to inverse mass.
- **Kinetic energy:** 0.5 Σ m v². Running mean displayed.

## Performance

- Pairwise checks are O(N²). Reduce `N` or `STEPS` for speed, or implement spatial binning.

## Reproducibility

- Set `SEED=None` for stochastic runs, integer for reproducibility.

## Troubleshooting

- **No window appears:** ensure a GUI backend (or X forwarding if remote).
- **Video save fails:** check FFmpeg installation and PATH.

## Files

- `particles.py` — main script
- `requirements.txt` — Python dependencies
