# low-fsl-mri

Bloch and microvascular diffusion--susceptibility simulation code for the paper.
The RF robustness comparison covers six preparations:
CW, composite, balanced, quadruple-refocused, sinusoidal, and CPMG-MLEV16.
This is not a complete reproduction of all paper results or experiments.

## Scope and access

This repository covers **simulation code only**: Bloch RF robustness and
synthetic microvascular diffusion--susceptibility simulations, including the
rate-fitting, cross-preparation prediction, and convergence workflows below.
In-vivo Pulseq generation, scanner acquisition, and acquired data are **out of
scope**, not planned deliverables or prerequisites for this simulation package.
The repository is currently private. No project license has been selected;
dependency licenses do not grant a license to this repository's own code.
No acquired image data, scanner sequence files, vendor SDK code, plotting
examples, internal project records, or source-history manifests are included.
Dependencies retain their own licenses; see `THIRD_PARTY_DEPENDENCIES.md`.

## Install and test

Tested with Python 3.13 and the dependency versions pinned below.

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

## RF robustness and Table 2

CW, composite, balanced, and quadruple-refocused preparations use the nominal
constant locking amplitude equal to each target: 20, 50, or 100 Hz. There is no
spectrum-peak alignment calibration. Sinusoidal RF has modulation frequency
`f` and peak amplitude `beta*f`, with `beta` the first positive root of `J0`.
CPMG retains timing frequency `1/(2*tau)=f` and 625-Hz composite refocusing RF;
it does not use a constant locking field equal to `f`.

The field grid is B0 = -250:5:250 Hz and B1 amplitude error = -25:0.5:25%.
Table 2 statistics use the full |B0| <= 250 Hz window, not the optional 100-Hz
window. RF cost here is not scanner-measured SAR.

```bash
python scripts/compare_nominal_table2.py
python scripts/compare_sinusoidal_table2.py --output sinusoidal_table2.csv
```

The first command compares the four piecewise-constant preparations and
finite-pulse CPMG (with the Foltz fit-time correction) and 160-ms spectral
peaks. Its sinusoidal values are reference-only; use the second command to
actually recompute all three sinusoidal rows.

`sinusoidal_continuous.py` evaluates the analytic cosine RF at every RK4 stage.
The default maximum integration step is 25 microseconds, adjusted to an integer
number of steps per period. This is continuous-waveform numerical integration,
not an exact analytic Bloch solution. It starts at +Mx, uses T1=1.2 s and
T2=80 ms, samples [0,2,4,8]/f seconds, and fits |Mx| with no Foltz correction.
The separate `engine.py` sinusoidal builder still uses midpoint-sampled,
piecewise-constant 25-microsecond RF; it is not silently substituted for the
continuous integration.

The full-grid continuous-integration rerun matches Table 2 to the reported
two-decimal precision:

| Target Hz | Median absolute error % | 95th percentile % |
|---:|---:|---:|
| 20 | 38.31 | 80.15 |
| 50 | 45.80 | 95.56 |
| 100 | 47.35 | 111.57 |

This verifies those three statistics, not every figure or acquisition result.

## Synthetic diffusion--susceptibility simulation

`microvascular.py` contains the straight-cylinder FFT susceptibility field,
RK4 Bloch update, periodic nearest-neighbor diffusion with no flux across the
vessel wall, and ideal CW/composite/balanced/CPMG preparations. Frequencies
are converted from Hz to rad/s explicitly. Axial reduction is allowed only
when both the field and water mask are y-invariant.

Paper defaults are: radius 6 micrometers, volume fraction 0.035, susceptibility
difference 0.18 ppm, B0 = 3 T, D = 0.8 square micrometers/ms, T1 = 1.3 s,
T2 = 80 ms, 64-cubed reference geometry (axially reduced when the field is
y-invariant), and a 12.5-microsecond reference step, with the period grid
chosen in multiples of four so ideal pulse centers are representable. The
paper frequency schedule is every integer from 1 to 100 Hz, not a sparser
subset. No acquired geometry or stored pilot rate table is used.

```bash
# Small end-to-end check only; not publication-resolution results:
python scripts/run_microvascular.py --grid 8 --dt-us 50 --frequencies 20,50,100

# Paper-resolution sweep (computationally substantial):
python scripts/run_microvascular_paper.py --frequencies paper --output microvascular_rates

# Fit both Lorentzian models to CW rates only, then predict the other preparations:
python scripts/fit_lorentzian_transfer.py --input microvascular_rates/fits.csv --output transfer_fit

# Paper convergence subset: 10, 20, and 50 Hz; 64^3 at 12.5 us vs 6.25 us vs 96^3:
python scripts/check_microvascular_convergence.py --output microvascular_convergence
```

Ordinary/composite/balanced use 13 durations up to min(0.2 s, 4/f), snapped
to representable ideal pulse centers. CPMG samples completed periods, with
up to 3 periods at or below 11 Hz and min(5, floor(0.25*f)) periods above.
Fit rejection requires positive finite samples >=1e-12 and R-squared >=0.95.
These are ideal instantaneous-pulse simulations; they are separate from the
finite-pulse Table 2 CPMG. The ideal CPMG modulation is phase-equivalent for
this model; it does not implement finite-width MLEV16 phase cycling.

The uncorrected Lorentzian is `R(f) = R_bg + A / (1 + (2 pi f tau)^2)`, fitted
to the 100 CW points only. The corrected model evaluates
`S(T) = exp(-R_bg T - A' G(T, tau))` with the same causal-filter quadrature
and Gauss-Newton monoexponential refit used for the paper tables, then predicts
the other preparations without refitting. Metrics are reported over 1-100 Hz
and 20-100 Hz. PDEV is the fraction of simulated preparation-minus-CW rate
differences explained by the model; the uncorrected model has PDEV = 0 by
construction. In this package, that pooled PDEV is taken over composite,
balanced, and ideal-pulse CPMG. This differs from a balanced/CPMG/sinusoidal
pool that excludes composite. Therefore a numerical match to the paper's
94.8% Delta-R figure should not be assumed even after a full rerun; the
pool definitions must be reconciled before comparing that statistic.

Kernel checks, a small-grid driver, and a small-grid
simulation-to-fit/convergence path have been validated. The full 1-100 Hz,
64-cubed rates, the 10/20/50 Hz 64^3 / 6.25 us / 96^3 convergence subset, and
the paper Lorentzian parameter and prediction tables have NOT yet been
validated as matching the paper at full resolution.
These commands recompute the simulation; they do not load previously stored
rate tables. Do not interpret the presence of the pipeline as reproduction of
those paper tables.

## Limitations and scope boundary

In-vivo Pulseq generators, `.seq` files, human or phantom acquisitions, and
vendor scanner code are not distributed or promised by this repository.

Tests check numerical consistency, conservation, waveform definitions,
schedule formulas, Lorentzian limiting cases, and selected small-grid
pipeline paths. Passing tests is not scanner safety, clinical validation, or
complete paper replication. The numerical reproduction limitations described
above apply independently of acquisition code, which is out of scope.
