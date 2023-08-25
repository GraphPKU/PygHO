import unittest
from pygho import MaskedTensor, SparseTensor
import pygho.backend.MaTensor as MaTensor
import pygho.backend.Mamamm as Mamamm
import pygho.backend.Spmamm as Spmamm
import torch


EPS = 1e-5

def maxdiff(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.amax((a-b).abs()).item()

def tensorequal(a: torch.Tensor, b: torch.Tensor) -> bool:
    return torch.all(a==b).item()

def floattensorequal(a: torch.Tensor, b: torch.Tensor) -> bool:
    return maxdiff(a, b) < EPS

class MaTensorTest(unittest.TestCase):
    def setUp(self) -> None:
        B = 2
        N = 3
        L = 5
        M = 7
        data = torch.randn((B, N, M, L))
        mask = torch.ones((B, N, M), dtype=torch.bool)
        mask[0, 2:] = False
        mask[0, :, 2:] = False
        mask[1, 1:] = False
        mask[1, :, 1:] = False
        vd = torch.masked.masked_tensor(data, mask.unsqueeze(-1).expand_as(data))
        mt = MaskedTensor(data, mask, padvalue=torch.inf)
        self.data = data
        self.mask = mask
        self.vd = vd
        self.mt = mt
        return super().setUp()
    
    def test_filterinf(self):
        A = torch.tensor([-torch.inf, 0, torch.inf, 1, 2, -torch.inf, 3])
        self.assertTrue(tensorequal(MaTensor.filterinf(A), torch.tensor([0, 0, 0, 1, 2, 0, 3])), "filter inf error")

    def test_fill(self):
        self.assertTrue(floattensorequal(self.mt.fill_masked(1024), torch.masked_fill(self.data, torch.logical_not(self.mask).unsqueeze(-1).expand_as(self.data), 1024)), "mask fill error")

    def test_pool(self):
        self.assertTrue(floattensorequal(self.mt.max(dim=1), self.vd.amax(dim=1).to_dense()), "max error")
        self.assertTrue(floattensorequal(self.mt.min(dim=1), self.vd.amin(dim=1).to_dense()), "min error")
        self.assertTrue(floattensorequal(self.mt.mean(dim=1), self.vd.mean(dim=1).to_dense()), "mean error")
        self.assertTrue(floattensorequal(self.mt.sum(dim=1), self.vd.sum(dim=1).to_dense()), "sum error")

        self.assertTrue(floattensorequal(self.mt.max(), self.vd.amax().to_dense()), "max error")
        self.assertTrue(floattensorequal(self.mt.min(), self.vd.amin().to_dense()), "min error")
        self.assertTrue(floattensorequal(self.mt.mean(), self.vd.mean().to_dense()), "mean error")
        self.assertTrue(floattensorequal(self.mt.sum(), self.vd.sum().to_dense()), "sum error")


class SpmammTest(unittest.TestCase):
    def setUp(self) -> None:
        b, n, m, l, d = 5, 3, 7, 13, 11
        A = torch.rand((b, n, m, d))
        Amask = torch.rand_like(A[:, :, :, 0]) > 0.9
        MA = MaskedTensor(A, Amask)
        ind = Amask.to_sparse_coo().indices()
        SA = SparseTensor(ind, A[ind[0], ind[1], ind[2]], shape=MA.shape)
        B = torch.rand((b, m, l, d))
        Bmask = torch.rand_like(B[:, :, :, 0]) > 0.9
        MB = MaskedTensor(B, Bmask)
        ind = Bmask.to_sparse_coo().indices()
        SB = SparseTensor(ind, B[ind[0], ind[1], ind[2]], shape=MB.shape)
        mask = torch.ones((b, n, l), dtype=torch.bool)
        self.SA = SA
        self.MB = MB
        self.mask = mask
        self.MA = MA
        self.SB = SB
        return super().setUp()
    
    def test_spmamm(self):
        self.assertTrue(floattensorequal(Spmamm.spmamm(self.SA, self.MB, self.mask).data, torch.einsum("bnmd,bmld->bnld", self.MA.data, self.MB.data)), "spmamm error")

    def test_maspmm(self):
        self.assertTrue(floattensorequal(Spmamm.maspmm(self.MA, self.SB, self.mask).data, torch.einsum("bnmd,bmld->bnld", self.MA.data, self.MB.data)), "maspmm error")
