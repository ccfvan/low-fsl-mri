"""Synthetic straight-cylinder Bloch--Torrey kernel; ideal refocusing pulses.

All precession arguments passed to Bloch propagation are angular frequencies.
Only a y-invariant straight cylinder permits the optional axial reduction.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
import numpy as np
from scipy import sparse

@dataclass
class Params:
    nx: int = 64
    ny: int = 64
    nz: int = 64
    m0: float = 1.0
    dt_s: float = 12.5e-6
    gamma_hz_t: float = 42.57747892e6
    remove_mean_offset: bool = True
    radius_um: float = 6.0
    volume_fraction: float = 0.035
    b0_t: float = 3.0
    t1_s: float = 1.3
    t2_s: float = 0.08
    d_um2_ms: float = 0.8
    delta_chi: float = 0.18e-6

    @property
    def r1_s_1(self): return 1/self.t1_s
    @property
    def r2_s_1(self): return 1/self.t2_s
    @property
    def d_um2_s(self): return self.d_um2_ms*1000

def periodic_displacement(d: np.ndarray, period: float) -> np.ndarray:
    return np.mod(d + period / 2.0, period) - period / 2.0


def fft_frequency_vector(n: int, d_um: float) -> np.ndarray:
    if n % 2 == 0:
        freq = np.r_[0:n // 2, -n // 2:0] / (n * d_um)
    else:
        freq = np.r_[0:(n // 2 + 1), -(n // 2):0] / (n * d_um)
    return 2.0 * np.pi * freq


def make_fft_cylinder_field(p: Params) -> dict[str, np.ndarray | float]:
    length_um = math.sqrt(math.pi * p.radius_um ** 2 / p.volume_fraction)
    dx_um = length_um / p.nx
    dy_um = length_um / p.ny
    dz_um = length_um / p.nz

    x = np.arange(p.nx) * dx_um
    y = np.arange(p.ny) * dy_um
    z = np.arange(p.nz) * dz_um
    xg, yg, zg = np.meshgrid(x, y, z, indexing="xy")

    center_x = 0.5 * length_um
    center_z = 0.5 * length_um
    ddx = periodic_displacement(xg - center_x, length_um)
    ddz = periodic_displacement(zg - center_z, length_um)
    mask = np.sqrt(ddx ** 2 + ddz ** 2) <= p.radius_um
    water = ~mask

    chi_map = np.zeros(mask.shape, dtype=float)
    chi_map[mask] = p.delta_chi

    kx = fft_frequency_vector(p.nx, dx_um)
    ky = fft_frequency_vector(p.ny, dy_um)
    kz = fft_frequency_vector(p.nz, dz_um)
    kxg, kyg, kzg = np.meshgrid(kx, ky, kz, indexing="xy")
    k2 = kxg ** 2 + kyg ** 2 + kzg ** 2
    kernel = np.zeros_like(k2)
    use = k2 > 0
    kernel[use] = 1.0 / 3.0 - (kzg[use] ** 2 / k2[use])

    delta_b_t = p.b0_t * np.fft.ifftn(np.fft.fftn(chi_map) * kernel).real
    delta_offset_hz = p.gamma_hz_t * delta_b_t
    removed_mean = 0.0
    if p.remove_mean_offset and np.any(water):
        removed_mean = float(delta_offset_hz[water].mean())
        delta_offset_hz[water] -= removed_mean
        delta_b_t[water] -= removed_mean / p.gamma_hz_t
    delta_offset_hz[mask] = 0.0
    delta_b_t[mask] = 0.0

    return {
        "delta_offset_hz": delta_offset_hz,
        "delta_b_t": delta_b_t,
        "water_mask": water,
        "inclusion_mask": mask,
        "length_um": length_um,
        "dx_um": dx_um,
        "dy_um": dy_um,
        "dz_um": dz_um,
        "actual_volume_fraction": float(mask.mean()),
        "removed_mean_offset_hz": removed_mean,
    }


def diffusion_alphas(p: Params, field: dict[str, np.ndarray | float]) -> tuple[float, float, float]:
    ax = p.d_um2_s * p.dt_s / (field["dx_um"] ** 2)
    ay = p.d_um2_s * p.dt_s / (field["dy_um"] ** 2)
    az = p.d_um2_s * p.dt_s / (field["dz_um"] ** 2)
    if ax + ay + az > 0.49:
        raise ValueError(f"3D diffusion step may be unstable: alpha sum={ax + ay + az:.4g}")
    return float(ax), float(ay), float(az)


def build_transition_matrix(water_mask: np.ndarray, ax: float, ay: float, az: float) -> tuple[sparse.csr_matrix, np.ndarray]:
    water_flat = water_mask.ravel()
    water_idx = np.flatnonzero(water_flat)
    nwater = water_idx.size
    index_map = np.full(water_mask.size, -1, dtype=np.int64)
    index_map[water_idx] = np.arange(nwater, dtype=np.int64)
    index_map = index_map.reshape(water_mask.shape)

    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    vals: list[np.ndarray] = []
    self_weights = np.ones(nwater, dtype=float)
    shifts = [
        ((0, 1, 0), ax), ((0, -1, 0), ax),
        ((1, 0, 0), ay), ((-1, 0, 0), ay),
        ((0, 0, 1), az), ((0, 0, -1), az),
    ]
    for shift, alpha in shifts:
        if alpha == 0:
            continue
        neighbor_water = np.roll(water_mask, shift, axis=(0, 1, 2))
        source_map = np.roll(index_map, shift, axis=(0, 1, 2))
        use = water_mask & neighbor_water
        dest = index_map[use]
        src = source_map[use]
        rows.append(dest)
        cols.append(src)
        vals.append(np.full(dest.shape, alpha, dtype=float))
        self_weights[dest] -= alpha

    rows.append(np.arange(nwater, dtype=np.int64))
    cols.append(np.arange(nwater, dtype=np.int64))
    vals.append(self_weights)
    mat = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(nwater, nwater))
    return mat, water_idx


def bloch_rhs(
    mx: np.ndarray, my: np.ndarray, mz: np.ndarray, dw: np.ndarray,
    omega1x: float, omega1y: float, p: Params
):
    dmx = -p.r2_s_1 * mx + dw * my - omega1y * mz
    dmy = -dw * mx - p.r2_s_1 * my + omega1x * mz
    dmz = omega1y * mx - omega1x * my - p.r1_s_1 * mz + p.r1_s_1 * p.m0
    return dmx, dmy, dmz


def rk4_step(
    mx, my, mz, dw, omega1x, p: Params, water_mask: np.ndarray,
    omega1y: float = 0.0, dt_s: float | None = None
):
    dt = p.dt_s if dt_s is None else float(dt_s)
    k1 = bloch_rhs(mx, my, mz, dw, omega1x, omega1y, p)
    k2 = bloch_rhs(mx + 0.5 * dt * k1[0], my + 0.5 * dt * k1[1], mz + 0.5 * dt * k1[2], dw, omega1x, omega1y, p)
    k3 = bloch_rhs(mx + 0.5 * dt * k2[0], my + 0.5 * dt * k2[1], mz + 0.5 * dt * k2[2], dw, omega1x, omega1y, p)
    k4 = bloch_rhs(mx + dt * k3[0], my + dt * k3[1], mz + dt * k3[2], dw, omega1x, omega1y, p)
    mx = mx + (dt / 6.0) * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
    my = my + (dt / 6.0) * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
    mz = mz + (dt / 6.0) * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
    mx[~water_mask] = 0.0
    my[~water_mask] = 0.0
    mz[~water_mask] = 0.0
    return mx, my, mz


def apply_transition(m: np.ndarray, transition: sparse.csr_matrix, water_idx: np.ndarray) -> np.ndarray:
    flat = np.zeros(m.size, dtype=float)
    flat[water_idx] = transition @ m.ravel()[water_idx]
    return flat.reshape(m.shape)


class Evolution:
    def __init__(self, p, field, axial=False):
        self.p = p
        water = field['water_mask']
        offset = field['delta_offset_hz']
        ax, ay, az = diffusion_alphas(p, field)  # Retain original 3D CFL check.
        if axial:
            if not np.all(water == water[:1]):
                raise ValueError('Axial reduction requires a y-invariant water mask')
            if not np.allclose(offset, offset[:1], rtol=0, atol=1e-12):
                raise ValueError('Axial reduction requires a y-invariant field')
            water = water[:1].copy()
            offset = offset[:1].copy()
            ay = 0.  # y diffusion vanishes identically on a y-invariant state.
        self.water = water
        self.dw = offset * (2*np.pi)
        self.transition, self.indices = build_transition_matrix(water, ax, ay, az)

    def samples(self, f, method, times):
        """Yield (TSL, Mx) along one evolution; balanced accepts one TSL only."""
        times = np.asarray(times, dtype=float)
        if not len(times):
            return
        if np.any(times < 0) or np.any(np.diff(times) <= 0):
            raise ValueError('TSLs must be nonnegative, strictly increasing')
        if method not in ('ordinary', 'balanced', 'cpmg', 'composite'):
            raise ValueError('Unknown preparation')
        if method in ('balanced', 'composite') and len(times) != 1:
            raise ValueError('Balanced/composite timing depends on total TSL; use independent evolutions')
        p = self.p
        water = self.water
        steps_sample = np.rint(times/p.dt_s).astype(int)
        if np.any(np.abs(steps_sample*p.dt_s-times) > 1e-9):
            raise ValueError('TSL cannot be represented exactly at dt')
        if len(set(steps_sample)) != len(times):
            raise ValueError('Duplicate integration-grid sample')
        mx = np.where(water, 1., 0.)
        my = np.zeros_like(mx)
        mz = np.zeros_like(mx)
        T = float(times[-1])
        pulses = []
        if method == 'balanced':
            pulses = [(T/4, np.pi), (3*T/4, -np.pi)]
        elif method == 'composite' and T > 0:
            # Witschey 2007 composite SL: SL(+x) T/2, 180 refocusing about the lock axis, SL(-x) T/2.
            pulses = [(T/2, np.pi)]
        elif method == 'cpmg' and T > 0:
            half = 1/(2*f)
            count = int(round(T/half))
            if count < 1 or abs(count*half-T) >= 1e-9:
                raise ValueError('CPMG requires complete echo cycles')
            pulses = [((k+.5)*half, np.pi) for k in range(count)]
        pulse_steps = {round(t/p.dt_s): angle for t, angle in pulses}
        targets = dict(zip(steps_sample, times))
        if 0 in targets:
            yield float(targets[0]), 1.
        for j in range(int(steps_sample[-1])):
            if j in pulse_steps:
                angle = pulse_steps[j]
                c, s = math.cos(angle), math.sin(angle)
                my, mz = c*my-s*mz, s*my+c*mz
            if method == 'ordinary':
                rf = f
            elif method in ('composite',):
                # Composite preparation: reverse lock polarity after the central ideal pi pulse.
                rf = f if j*p.dt_s < T/2 else -f
            elif method == 'balanced':
                rf = f if j*p.dt_s < T/4 or j*p.dt_s >= 3*T/4 else -f
            else:
                rf = 0.
            mx, my, mz = rk4_step(mx, my, mz, self.dw, 2*np.pi*rf, p, water)
            mx = apply_transition(mx, self.transition, self.indices)
            my = apply_transition(my, self.transition, self.indices)
            mz = apply_transition(mz, self.transition, self.indices)
            if j+1 in targets:
                yield float(targets[j+1]), float(mx[water].mean())

    def independent(self, f, method, time):
        return next(self.samples(f, method, [time]))[1]
