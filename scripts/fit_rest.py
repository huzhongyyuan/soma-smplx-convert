"""Fit SMPL-X to the SOMA rest mesh once; writes the shape + surface attachments used by MeshConverter.
Usage: python scripts/fit_rest.py --smplx-model SMPLX_NEUTRAL_2020.npz --out rest_fit.npz"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from soma_smplx import meshfit

ap = argparse.ArgumentParser()
ap.add_argument('--smplx-model', required=True); ap.add_argument('--out', default='rest_fit.npz')
a = ap.parse_args()
meshfit.fit_rest(a.smplx_model, a.out)
