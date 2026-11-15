# low-fsl-mri

Simulation code for the six magnetization-prepared relaxation methods compared in the paper *Robust Magnetization-Prepared Relaxation Mapping*:

1. conventional continuous-wave spin lock
2. composite spin lock (internal key `rotary_echo`)
3. balanced spin lock
4. quadruple-refocused spin lock
5. sinusoidal spin lock
6. CPMG-MLEV16

The package builds preparation waveforms, propagates a piecewise-constant Bloch model, computes finite-duration spectra, fits a monoexponential relaxation time, applies the Foltz timing correction for CPMG, and reports RF amplitude metrics. Triple-refocused spin lock can still be constructed, but it is not part of the paper comparison set.

This is a method-implementation package. It is not a complete reproduction of the paper's numerical results, figures, or phantom and in vivo measurements. Scanner sequences and acquired image data are not included.

A license for this code has not been selected.

## Requirements

Runtime dependencies are pinned in `requirements.txt`:

- numpy
- scipy
- pandas

Install with:

```bash
python -m pip install -r requirements.txt
```

License information for those third-party packages is summarized in `THIRD_PARTY_DEPENDENCIES.md`. Those licenses apply to the packages themselves, not to the code in this repository.

## Tests

The `tests/` directory contains automated checks of the simulation kernels. These tests do not plot figures and do not reproduce the paper figures. They check the six-method set, preparation durations, default field-grid sampling, the two candidate |B0| summary windows, pulse counts and phase tables, nominal on-resonance decay, the sinusoidal waveform definition, spectrum normalization, the Foltz correction, and RF-cost differences between methods.

Run them from the repository root:

```bash
python -m unittest discover -s tests -v
```

## Nominal locking amplitudes

CW, composite, balanced, and quadruple-refocused preparations use a constant locking amplitude equal to the target frequency: 20, 50, or 100 Hz. `build_core` applies that amplitude directly. The package does not peak-align the lock.

CPMG-MLEV16 keeps the timing frequency `1/(2*tau) = target` and 625 Hz composite refocusing pulses. Sinusoidal spin lock keeps modulation frequency equal to the target and peak amplitude `beta * target`, with `beta` the first positive root of `J0`.

The numerical comparison with the paper's Table 2, over `|B0| <= 250 Hz` and `|B1| <= 25%`, and the 160 ms spectral peaks is:

```bash
python scripts/compare_nominal_table2.py
```

That command reruns CW, composite, balanced, quadruple-refocused, and CPMG-MLEV16. It does not rerun the sinusoidal Bloch grid: the published sinusoidal medians used a separate continuous-waveform integration, while this package samples the same analytic sinusoid as piecewise-constant steps.

## Default settings and limits

- The default field grid samples B0 from -250 Hz to 250 Hz in 5 Hz steps, and B1 from -25% to 25% in 0.5% steps.
- Summary windows of |B0| <= 100 Hz and |B0| <= 250 Hz are both reported. Neither window is selected as the paper statistic.
- The default sinusoidal sampling step is 25 microseconds. This can be overridden through `Params.sinusoidal_dt_s`.
- Reported RF metrics are computed from the simulated waveforms. They are not scanner-measured SAR.

Passing the included tests means those kernel checks succeeded. It does not mean that all paper figures or tabulated results have been reproduced.
