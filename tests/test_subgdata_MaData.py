import unittest
from pygho import MaskedTensor, SparseTensor
import pygho.backend.MaTensor as MaTensor
import pygho.backend.Mamamm as Mamamm
import pygho.backend.Spmamm as Spmamm
import torch


EPS = 1e-5

def maxdiff(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.max((a-b).abs()).item()

def tensorequal(a: torch.Tensor, b: torch.Tensor) -> bool:
    return torch.all(a==b).item()

def floattensorequal(a: torch.Tensor, b: torch.Tensor) -> bool:
    return maxdiff(a, b) < EPS
