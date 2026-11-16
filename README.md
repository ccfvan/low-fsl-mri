# low-fsl-mri

Candidate simulation package for the six RF preparations compared in the paper:
CW, composite, balanced, quadruple-refocused, sinusoidal, and CPMG-MLEV16.
This is not a complete reproduction of all paper results or experiments.

## Scope and access

The release is being prepared privately. Public release and the project license
still require the authors' approval. No license is selected by this draft.
No acquired image data, scanner sequence files, vendor SDK code, plotting
examples, internal project records, or source-history manifests are included.
Dependencies retain their own licenses; see `THIRD_PARTY_DEPENDENCIES.md`.

## Install and test

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

The driver defaults to radius 6 micrometers, volume fraction 0.035,
susceptibility difference 0.18 ppm, B0=3 T, D=0.8 square micrometers/ms,
T1=1.3 s, T2=80 ms, 64-cubed reference geometry, and a maximum 12.5-microsecond
step. The period grid is chosen in multiples of four so ideal pulse centers
are representable. No acquired geometry or stored pilot rate table is used.

```bash
# Small end-to-end check only; not publication-resolution results:
python scripts/run_microvascular.py --grid 8 --dt-us 50 --frequencies 20,50,100
# Computationally substantial paper-geometry run:
python scripts/run_microvascular.py --frequencies 1:100
```

Ordinary/composite/balanced use 13 durations up to min(0.2 s,4/f).
CPMG samples completed periods, with up to 3 periods below/equal 11 Hz and
up to min(5,floor(0.25*f)) above. Fit rejection requires positive finite
samples >=1e-12 and R-squared >=0.95. These are ideal instantaneous-pulse
simulations; they are separate from finite-pulse Table 2 CPMG. The ideal
CPMG modulation is phase-equivalent for this model; it does not implement
finite-width MLEV16 phase cycling.

Only kernel checks and a small-grid driver run have been validated in this
candidate. The full 1--100 Hz, 64-cubed rates, 96-cubed/time-step convergence,
and the corrected/uncorrected Lorentzian cross-preparation prediction tables
have NOT been rerun here. The Lorentzian prediction/fitting entry is not yet
included. Do not interpret this candidate as reproducing those paper tables.

## Pulseq and remaining gaps

An independently rights-cleared, self-contained in-vivo Pulseq generator is
not included. The acquisition chain still requires separation of author-written
T1rho scheduling from other sequence code and review of code ownership and
necessary copyright notices. No `.seq` batch or acquired data is redistributed.

If a sequence generator is added later, MATLAB Pulseq or PyPulseq must be
installed as a versioned external dependency under its own original license;
its source must not be presented as this project's original code. No exact
Pulseq dependency version has been certified by this candidate.

Tests check numerical consistency, conservation, waveform definitions and
selected limiting cases. Passing tests is not scanner safety, clinical
validation, or complete paper replication. Public release remains pending
license/rights approval and completion of the stated gaps.
