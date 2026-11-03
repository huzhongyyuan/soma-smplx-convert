"""Asset-free mathematical checks for the body solver (requires ARDY imports)."""
import unittest
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from soma_smplx.body import Attachment, solve_lm, reference


class BodyNumerics(unittest.TestCase):
    def test_attachment_normal_offset_derivative(self):
        gen = torch.Generator().manual_seed(90411)
        vertices = torch.randn(1, 6, 3, generator=gen, dtype=torch.float64)
        tangent = torch.randn(1, 6, 3, 4, generator=gen, dtype=torch.float64)
        att = Attachment(np.array([[0,1,2],[3,4,5]]),
                         (np.arange(2), np.array([[.2,.3,.5],[.3,.4,.3]]), np.array([.04,-.02])))
        _, jac = att.evaluate(vertices, tangent)
        eps = 1e-6
        fd = torch.stack([(att.evaluate(vertices+eps*tangent[...,k])-att.evaluate(vertices-eps*tangent[...,k]))/(2*eps) for k in range(4)], -1)
        self.assertLess(float((fd-jac).abs().max()), 1e-8)

    def test_degenerate_attachment_is_rejected(self):
        att = Attachment(np.array([[0,1,2]]), (np.array([0]),np.array([[.2,.3,.5]]),np.array([0.])))
        with self.assertRaisesRegex(ValueError, 'Degenerate'):
            att.evaluate(torch.zeros(1,3,3))

    def test_rotation_exponential_against_scipy(self):
        vectors = np.array([[0,0,0],[1e-10,0,0],[.2,-.4,.5],[np.pi-1e-7,0,0]],dtype=np.float64)
        actual = reference.expm(torch.from_numpy(vectors)).numpy()
        np.testing.assert_allclose(actual, Rotation.from_rotvec(vectors).as_matrix(), atol=1e-12, rtol=0)

    def test_independent_batched_lm_solutions(self):
        target = torch.tensor([[0.,0.],[3.,-4.],[-.5,10.]],dtype=torch.float64)
        matrix = torch.tensor([[2.,.1],[.1,4.],[1.,-2.]],dtype=torch.float64)
        residual = lambda state: (state[0]-target)@matrix.T
        jac = lambda state: matrix.expand(3,3,2)
        result, _ = solve_lm((torch.zeros_like(target),),lambda state,delta:(state[0]+delta,),residual,jac)
        np.testing.assert_allclose(result[0].numpy(),target.numpy(),atol=1e-6,rtol=0)


if __name__ == '__main__':
    unittest.main()
