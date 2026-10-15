"""Synthetic mathematical checks; no experimental input, no CALPHAD."""
import importlib.util
from pathlib import Path
import numpy as np
from d_common import project,tvd
s=importlib.util.spec_from_file_location('analyze',Path(__file__).with_name('01_analyze.py'));m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def run():
    y=np.array([[1.,0],[1,0],[0,1],[0,1]])
    t,b,w=m.variance(y,['A','A','B','B']);assert np.allclose(t,b) and np.allclose(w,0)
    t,b,w=m.variance(y,['A','B','A','B']);assert np.allclose(b,0) and np.allclose(t,w)
    out=project([[-2,-1],[2,2]],[.2,.8]);assert np.allclose(out,[[.2,.8],[.5,.5]])
    assert np.allclose(tvd([[1,0]],[[0,1]]),1)
    dist,n,complete=m.transitions(np.array([0,1,2,3]),np.array([0,0,1,1]),np.arange(4));assert np.allclose(dist,[1.5,.5,.5,1.5]) and n==1 and complete
    dist,n,complete=m.transitions(np.array([0,2,3]),np.array([0,1,1]),np.arange(4));assert n==0 and np.isinf(dist).all() and not complete
    tr=np.array([[0,0,0],[1,0,0],[0,1,0],[1,1,0],[.5,.5,0.]])
    *_,diag=m.manifold(tr,np.array([.5,.5,0]));assert diag['affine_rank']==2 and diag['convex_reconstruction_error']<1e-8
    *_,diag=m.manifold(tr,np.array([.5,.5,2.]));assert abs(diag['affine_residual']-2)<1e-8
    print('PASS: variance identities, topology transitions with missing-grid gaps, simplex fallback, TVD, affine and convex geometry')
if __name__=='__main__':run()
