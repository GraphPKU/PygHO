import unittest
from pygho import SparseTensor
import pygho.backend.SpTensor as SpTensor
import pygho.backend.Spspmm as Spspmm
import pygho.backend.Spmm as Spmm
import torch
import numpy as np


def maxdiff(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.max((a - b).abs()).item()


def tensorequal(a: torch.Tensor, b: torch.Tensor) -> bool:
    return torch.all(a == b).item()


def lexsort(keys, dim: int = -1):
    '''
    lexsort ascendingly
    '''
    tmpkey = torch.flip(keys, dims=(0, ))
    ind = np.lexsort(tmpkey.detach().cpu().numpy(), axis=dim)
    return keys[:, torch.from_numpy(ind).to(keys.device)]


EPS = 5e-5


class SpTensorTest(unittest.TestCase):

    def setUp(self) -> None:
        return super().setUp()

    def test_hash_tight(self):
        sd, sshape, nnz, d = 5, (2, 3, 7, 11, 13), 17, 7
        indices = torch.stack(
            tuple(torch.randint(sshape[i], (nnz, )) for i in range(sd)))
        tsshape = torch.LongTensor(sshape)
        thash = SpTensor.indicehash_tight(indices, tsshape)
        hhash = (((
            (indices[0]) * sshape[1] + indices[1]) * sshape[2] + indices[2]) *
                 sshape[3] + indices[3]) * sshape[4] + indices[4]
        self.assertTrue(tensorequal(thash, hhash), "hash_tight wrong")
        dthash = SpTensor.decodehash_tight(thash, tsshape)
        self.assertTrue(tensorequal(dthash, indices),
                        "hash_tight decode wrong")

    def test_hash(self):
        sd, sshape, nnz, d = 5, (2, 3, 7, 11, 13), 17, 7
        indices = torch.stack(
            tuple(torch.randint(sshape[i], (nnz, )) for i in range(sd)))
        indices = lexsort(indices, dim=-1)
        thash = SpTensor.indicehash(indices)
        self.assertTrue(torch.all(torch.diff(thash) >= 0),
                        "hash not keep order")
        dthash = SpTensor.decodehash(thash, sparse_dim=len(sshape))
        self.assertTrue(tensorequal(dthash, indices),
                        "hash_tight decode wrong")

    def test_create(self):
        n, m, l, nnz, d = 2, 3, 5, 23, 7
        indices = torch.stack(
            (torch.randint(0, n, (nnz, )), torch.randint(0, m, (nnz, )),
             torch.randint(0, l, (nnz, ))))
        values = torch.randn((nnz, d))

        A1 = torch.sparse_coo_tensor(indices, values, size=(n, m, l, d))
        A2 = SpTensor.SparseTensor(indices, values, (n, m, l, d), False)
        A2f = SpTensor.SparseTensor.from_torch_sparse_coo(A1)
        A1c = A1.coalesce()

        self.assertTrue(tensorequal(A2.indices, A1c.indices()),
                        "create indice not match")
        self.assertLessEqual(maxdiff(A2.values, A1c.values()), EPS,
                             "create value not match")

        self.assertTrue(tensorequal(A2.indices, A2f.indices),
                        "from coo indice not match")
        self.assertLessEqual(maxdiff(A2.values, A2f.values), EPS,
                             "from coo value not match")

        A1f = A2.to_torch_sparse_coo()
        self.assertLessEqual(maxdiff(A1f.to_dense(), A1c.to_dense()), EPS,
                             "to coo not match")


class SpspmmTest(unittest.TestCase):

    def setUp(self) -> None:

        return super().setUp()

    def test_ptr2batch(self):
        ptr = torch.tensor([0, 4, 4, 7, 8, 11, 11, 11, 16], dtype=torch.long)
        batch = torch.tensor([0, 0, 0, 0, 2, 2, 2, 3, 4, 4, 4, 7, 7, 7, 7, 7],
                             dtype=torch.long)
        self.assertTrue(tensorequal(Spspmm.ptr2batch(ptr, dim_size=16), batch),
                        "ptr2batch error")

    def test_2dmm(self):
        from torch_scatter import scatter_add
        n, m, l = 300, 200, 400
        device = torch.device("cuda")
        A = torch.rand((n, m), device=device)
        A[torch.rand_like(A) > 0.9] = 0
        A = A.to_sparse_coo()
        B = torch.rand((m, l), device=device)
        B[torch.rand_like(B) > 0.9] = 0
        B = B.to_sparse_coo()
        ind1 = A.indices()
        val1 = A.values()
        ind2 = B.indices()
        val2 = B.values()

        C = A @ B
        C = C.coalesce()

        ind, bcd = Spspmm.spspmm_ind(ind1, 1, ind2, 0)
        mult = val1[bcd[1]] * val2[bcd[2]]
        outval = scatter_add(mult, bcd[0], dim_size=ind.shape[1])
        out = torch.sparse_coo_tensor(ind, outval)
        out = out.coalesce()
        self.assertTrue(tensorequal(C.indices(), out.indices()),
                        "spspmm indice not match")
        self.assertLessEqual(maxdiff(C.values(), out.values()), EPS,
                             "spspmm value not match")

        tar_ind = torch.stack(
            (torch.randint_like(ind1[0], n), torch.randint_like(ind1[0], l)))

        tar_ind = SpTensor.decodehash(
            torch.unique(SpTensor.indicehash(tar_ind), sorted=True), 2)
        acd = Spspmm.filterind(tar_ind, ind, bcd)
        mult = val1[acd[1]] * val2[acd[2]]
        outval = scatter_add(mult, acd[0], dim_size=tar_ind.shape[1])
        maskedout = torch.sparse_coo_tensor(tar_ind, outval)
        maskedout = maskedout.coalesce()
        # debug spspmm with target filter
        self.assertLessEqual(
            maxdiff(maskedout.to_dense()[tar_ind[0], tar_ind[1]],
                    C.to_dense()[tar_ind[0], tar_ind[1]]), EPS,
            "spspmm with target ind value not match")

    def test_2dhadamard(self):
        n, m = 300, 200
        Ap = torch.rand((n, m))
        Ap[torch.rand_like(Ap) > 0.9] = 0
        Ap = Ap.to_sparse_coo()
        Bp = torch.rand((n, m))
        Bp[torch.rand_like(Bp) > 0.9] = 0
        Bp = Bp.to_sparse_coo()
        spsphadamardout = Spspmm.spsphadamard(
            SparseTensor.from_torch_sparse_coo(Ap),
            SparseTensor.from_torch_sparse_coo(Bp))
        self.assertLessEqual(
            torch.max((torch.multiply(Ap, Bp) -
                       spsphadamardout.to_torch_sparse_coo()
                       ).coalesce().values().abs()).item(), EPS,
            "hadamard error")

    def test_3dmm(self):
        from torch_scatter import scatter_add
        n, m, l, k = 13, 5, 7, 11
        A = torch.rand((n, k, m))
        A[torch.rand_like(A) > 0.5] = 0
        A = A.to_sparse_coo()
        B = torch.rand((l, k, n))
        B[torch.rand_like(B) > 0.5] = 0
        B = B.to_sparse_coo()
        ind1 = A.indices()
        val1 = A.values()
        ind2 = B.indices()
        val2 = B.values()

        C = torch.einsum("nkm,lkd->nmld", A.to_dense(), B.to_dense())
        Cs = C.to_sparse_coo().coalesce()

        ind, bcd = Spspmm.spspmm_ind(ind1, 1, ind2, 1)
        mult = val1[bcd[1]] * val2[bcd[2]]
        outval = scatter_add(mult, bcd[0], dim_size=ind.shape[1])
        out = torch.sparse_coo_tensor(ind, outval)
        out = out.coalesce()

        self.assertTrue(tensorequal(Cs.indices(), out.indices()),
                        "spspmm indice not match")
        self.assertLessEqual(maxdiff(Cs.values(), out.values()), 1e-5,
                             "spspmm value not match")

        tar_ind = torch.stack(
            (torch.randint_like(ind1[0], n), torch.randint_like(ind1[0], m),
             torch.randint_like(ind1[0], l), torch.randint_like(ind1[0], n)))

        tar_ind = SpTensor.decodehash(
            torch.unique(SpTensor.indicehash(tar_ind), sorted=True), 4)
        acd = Spspmm.filterind(tar_ind, ind, bcd)
        mult = val1[acd[1]] * val2[acd[2]]
        outval = scatter_add(mult, acd[0], dim_size=tar_ind.shape[1])
        maskedout = torch.sparse_coo_tensor(tar_ind, outval)
        maskedout = maskedout.coalesce()
        # debug spspmm with target filter
        self.assertLessEqual(
            maxdiff(
                maskedout.to_dense()[tar_ind[0], tar_ind[1], tar_ind[2],
                                     tar_ind[3]],
                C[tar_ind[0], tar_ind[1], tar_ind[2], tar_ind[3]]), 1e-5,
            "spspmm with target indice value not match")

    def test_3dhadamard(self):
        n, m, l = 3, 5, 11
        Ap = torch.rand((n, m, l))
        Ap[torch.rand_like(Ap) > 0.9] = 0
        Ap = Ap.to_sparse_coo()
        Bp = torch.rand((n, m, l))
        Bp[torch.rand_like(Bp) > 0.9] = 0
        Bp = Bp.to_sparse_coo()
        spsphadamardout = Spspmm.spsphadamard(
            SparseTensor.from_torch_sparse_coo(Ap),
            SparseTensor.from_torch_sparse_coo(Bp))
        self.assertLessEqual(
            torch.max((torch.multiply(Ap, Bp) -
                       spsphadamardout.to_torch_sparse_coo()
                       ).coalesce().values().abs()).item(), 1e-5,
            "3d hadamard value error")


class SpmmTest(unittest.TestCase):

    def setUp(self) -> None:
        return super().setUp()

    def test_spmm(self):
        n, m, l = 300, 200, 400
        device = torch.device("cuda")
        A = torch.rand((n, m), device=device)
        A[torch.rand_like(A) > 0.9] = 0
        A = A.to_sparse_coo()
        X = torch.randn((m, l), device=device)
        Y1 = Spmm.spmm(
            SparseTensor(A.indices(),
                         A.values().unsqueeze(-1), A.shape + (1, )), 1, X)
        Y2 = A @ X
        self.assertLessEqual(maxdiff(Y1, Y2), EPS, "spmm error")
    