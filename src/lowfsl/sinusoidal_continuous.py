"""Continuous sinusoidal Bloch integration, RF core only (no readout).

Classical RK4 evaluates the analytic RF at all four stage times; this is NOT
an exact analytic Bloch solution or the constant-RF propagator approximation.
RF peak = beta*f, modulation frequency = f; no spectrum-peak calibration.
"""
import numpy as np
from scipy.special import jn_zeros
BETA = float(jn_zeros(0, 1)[0])

def continuous_signals(f, b0, b1, dt=25e-6):
    """Integrate the existing affine Bloch equations with RK4, M(0)=+x."""
    if not np.isfinite(f) or f <= 0 or not np.isfinite(dt) or dt <= 0:
        raise ValueError("Frequency and integration step must be positive")
    b0, b1 = np.broadcast_arrays(np.asarray(b0, float), np.asarray(b1, float))
    if b0.ndim != 1 or not np.all(np.isfinite(b0)) or not np.all(np.isfinite(b1)):
        raise ValueError("B0 and B1 must be finite one-dimensional arrays")
    steps = int(np.ceil(1 / f / dt))
    h = 1 / f / steps
    m = np.zeros((3, b0.size)); m[0] = 1
    dw = 2*np.pi*b0
    def rhs(v, t):
        w = 2*np.pi*BETA*f*np.cos(2*np.pi*f*t)*b1
        x, y, z = v
        return np.array([-x/.08+dw*y, -dw*x-y/.08+w*z,
                         -w*y-z/1.2+1/1.2])
    signals = [m[0].copy()]
    for k in range(8*steps):
        t = k*h
        a = rhs(m,t); b = rhs(m+h*a/2,t+h/2)
        c = rhs(m+h*b/2,t+h/2); d = rhs(m+h*c,t+h)
        m += h*(a+2*b+2*c+d)/6
        if k+1 in (2*steps,4*steps,8*steps):
            signals.append(m[0].copy())
    return np.array(signals).T
