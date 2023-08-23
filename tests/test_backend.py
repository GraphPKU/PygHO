import unittest
from backend import SpTensor
import torch
    values = torch.randn((nnz, d))
    n, m, nnz, d = 5, 7, 17, 7
    indices = torch.stack(
        (torch.randint(0, n, (nnz, )), torch.randint(0, m, (nnz, ))))
    values = torch.randn((nnz, d))

    A1 = torch.sparse_coo_tensor(indices, values, size=(n, m, d))
    A2 = SpTensor.SparseTensor(indices, values, (n, m, d), False)
    A2f = SpTensor.SparseTensor.from_torch_sparse_coo(A1)

    print("debug from_torch_sparse_coo ", torch.max(
        (A2.indices - A2f.indices)), torch.max((A2.values - A2f.values)))

    A1c = A1.coalesce()
    print("debug coalesce ", torch.max((A2.indices - A1c.indices())),
          torch.max((A2.values - A1c.values())))

    A2cf = SpTensor.SparseTensor.from_torch_sparse_coo(A1c)
    print("debug from_torch_sparse_coo ", torch.max(
        (A2.indices - A2cf.indices)), torch.max((A2.values - A2cf.values)))

    A1t = A2.to_torch_sparse_coo()
    print("debug from_torch_sparse_coo ", (A1t - A1).coalesce())

    print("should of same shape and nnz ", A1c, A1t, A2, A2f, A2cf, sep="\n")

    # 3D SparseTensor
    n, m, l, nnz, d = 2, 3, 5, 73, 7
    indices = torch.stack(
        (torch.randint(0, n, (nnz, )), torch.randint(0, m, (nnz, )),
         torch.randint(0, l, (nnz, ))))
    values = torch.randn((nnz, d))

    A1 = torch.sparse_coo_tensor(indices, values, size=(n, m, l, d))
    A2 = SpTensor.SparseTensor(indices, values, (n, m, l, d), False)

    A2f = SpTensor.SparseTensor.from_torch_sparse_coo(A1)

    print("debug from_torch_sparse_coo ", torch.max(
        (A2.indices - A2f.indices)), torch.max((A2.values - A2f.values)))

    A1c = A1.coalesce()
    print("debug coalesce ", torch.max((A2.indices - A1c.indices())),
          torch.max((A2.values - A1c.values())))

    A2cf = SpTensor.SparseTensor.from_torch_sparse_coo(A1c)
    print("debug from_torch_sparse_coo ", torch.max(
        (A2.indices - A2cf.indices)), torch.max((A2.values - A2cf.values)))

    A1t = A2.to_torch_sparse_coo()
    print("debug from_torch_sparse_coo ",
          (A1t - A1).coalesce().values().abs().max())

    print("should of same shape and nnz ", A1c, A1t, A2, A2f, A2cf, sep="\n")
    A3 = A2.tuplewiseapply(lambda x: torch.ones_like(x))
    assert torch.all(A3.values == 1)
    print("debug tuplewiseapply ", A3)

    A2m = A2.max(dim=1, return_sparse=True)
    Adm = A2.max(dim=1)
    print(A2m.indices, A2m.values, Adm, A1.to_dense().max(dim=1)[0])
    A2mu = A2m.unpooling(dim=1, tarX=A2)
    print(A2mu.indices, A2mu.values)

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
        self.assertTrue(torch.all(thash == hhash), "hash_tight wrong")
        dthash = SpTensor.decodehash_tight(thash, tsshape)
        self.assertTrue(torch.all(dthash == indices), "hash_tight decode wrong")

    def test_hash(self):
        sd, sshape, nnz, d = 5, (2, 3, 7, 11, 13), 17, 7
        from torch_geometric.utils import lexsort
        indices = torch.stack(
            tuple(torch.randint(sshape[i], (nnz, )) for i in range(sd)))
        indices = lexsort(indices, dim=-1)
        tsshape = torch.LongTensor(sshape)
        thash = SpTensor.indicehash(indices)
        self.assertTrue(torch.all(torch.diff(thash)>=0), "hash not keep order")
        dthash = SpTensor.decodehash(thash, sparse_dim=len(sshape))
        self.assertTrue(torch.all(dthash == indices), "hash_tight decode wrong")
    
