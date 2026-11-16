"""Numerical kernel checks, not clinical or full-paper validation."""
import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lowfsl.sinusoidal_continuous import continuous_signals,BETA
from lowfsl import engine
from lowfsl.microvascular import Params,make_fft_cylinder_field,diffusion_alphas,build_transition_matrix,Evolution

class ContinuousSinusoidTests(unittest.TestCase):
    def test_nominal_decay_all_targets(self):
        for f in (20,50,100):
            y=continuous_signals(f,np.array([0.]),np.array([1.]))
            np.testing.assert_allclose(y[0],np.exp(-np.array([0,2,4,8])/f/.08),atol=1e-10)
    def test_step_halving(self):
        for f in (20,50,100):
            a=continuous_signals(f,np.array([-250.,100.,250.]),np.array([.75,1.,1.25]))
            b=continuous_signals(f,np.array([-250.,100.,250.]),np.array([.75,1.,1.25]),12.5e-6)
            np.testing.assert_allclose(a,b,rtol=0,atol=1e-5)
    def test_beta_and_invalid_input(self):
        self.assertAlmostEqual(BETA,2.4048255577,places=9)
        for f,dt in [(0,25e-6),(50,0),(-1,25e-6)]:
            with self.assertRaises(ValueError): continuous_signals(f,np.array([0]),np.array([1]),dt)
    def test_piecewise_convergence(self):
        f=100;tsl=2/f;b0=np.array([100.]);b1=np.array([1.1])
        reference=continuous_signals(f,b0,b1,6.25e-6)[0,1]
        errors=[]
        for dt in (100e-6,25e-6):
            p=engine.Params(sinusoidal_dt_s=dt)
            ops=engine.build_core(p,'sinusoidal',tsl,f)
            # Use the public constant-RF propagator directly on an initial +x state.
            value=engine.simulate_preparation(p,ops,b0,b1,{},initial_axis=0,readout_axis=0)[0]
            errors.append(abs(value-abs(reference)))
        self.assertLess(errors[1],errors[0])

class MicrovascularTests(unittest.TestCase):
    def test_diffusion_conserves_mass_and_constant_state(self):
        p=Params(nx=8,ny=8,nz=8);field=make_fft_cylinder_field(p)
        a=diffusion_alphas(p,field);matrix,idx=build_transition_matrix(field['water_mask'],*a)
        np.testing.assert_allclose(np.asarray(matrix.sum(axis=0)).ravel(),1,atol=1e-14)
        np.testing.assert_allclose(matrix @ np.ones(len(idx)),1,atol=1e-14)
    def test_zero_susceptibility_decay(self):
        p=Params(nx=8,ny=8,nz=8,delta_chi=0,dt_s=25e-6)
        e=Evolution(p,make_fft_cylinder_field(p),axial=True)
        for prep in ('ordinary','composite','balanced','cpmg'):
            self.assertAlmostEqual(e.independent(100,prep,.02),np.exp(-.02/.08),places=10)
    def test_axial_equals_full_3d(self):
        p=Params(nx=8,ny=8,nz=8,dt_s=25e-6);field=make_fft_cylinder_field(p)
        a,b=Evolution(p,field,axial=True),Evolution(p,field,axial=False)
        for prep in ('ordinary','composite','balanced','cpmg'):
            self.assertAlmostEqual(a.independent(100,prep,.01),b.independent(100,prep,.01),places=12)
    def test_axial_rejects_noninvariant_field(self):
        p=Params(nx=8,ny=8,nz=8);field=make_fft_cylinder_field(p)
        field['delta_offset_hz'][1,0,0]+=.1
        with self.assertRaises(ValueError): Evolution(p,field,axial=True)
    def test_cfl_and_off_grid_rejected(self):
        p=Params(nx=8,ny=8,nz=8,dt_s=1)
        with self.assertRaises(ValueError): diffusion_alphas(p,make_fft_cylinder_field(p))
        p.dt_s=25e-6;e=Evolution(p,make_fft_cylinder_field(p),axial=True)
        with self.assertRaises(ValueError): e.independent(100,'cpmg',.013)
        with self.assertRaises(ValueError): e.independent(100,'ordinary',.001001)
if __name__=='__main__':unittest.main()
