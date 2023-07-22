# pyg_subg

A library for subgraph GNN based on torch_geometric.

# Tested environments
```
Python 3.10.10
networkx==2.8.4
numpy==1.24.3
ogb==1.3.6
scikit_learn==1.2.2
scipy==1.10.1
torch==2.0.1
torch_geometric==2.3.0
torch_scatter==2.1.1+pt20cu118
torchmetrics==0.11.4
```
## Preliminary

Subgraph GNNs all use tuple representations $X\in \R^{n\times n\times d}$, where $X_{ij}$ is feature of node $j$ in subgraph $i$. $X$ can be sparse or dense. For simplicity, we neglect the $d$ dimension. We can build a subgraph identity matrix $B\in \{0,1\}^{n\times n}$, $B_{ik}=1$ means $k\in subg(j)$.

https://arxiv.org/pdf/2302.07090.pdf categorized subgraph GNNs' operators.

**Single Point Operation:** $Y_{ij}\leftarrow X_{ij}, X_{ii}, X_{jj}, X_{ji} \Rightarrow Y\leftarrow X, diag(X)11^T, 11^Tdiag(X), X^T$

**Global-Global Operation:** pooling on the whole graph
$$
Y_{ij}\leftarrow \sum_k X_{ik},\sum_k X_{kj}\Rightarrow Y\leftarrow X11^T, 11^TX
$$

**Global-Local Operation**: message passing on the whole graph
$$
Y_{ij}\leftarrow \sum_{k\in N(j, A)} X_{ik},\sum_{k\in N(i, A)} X_{kj} \Rightarrow
Y\leftarrow AX, XA
$$

**Local-Global Operation**: pooling with in subgraph. 
$$
Y_{ij}\leftarrow \sum_{k\in subg(i)} X_{ik}\Rightarrow Y\leftarrow (X\odot B)11^T
$$
**Local-Local Operation**: message passing in each subgraph
for induced subgraphs
$$
Y_{ij}\leftarrow \sum_{k\in N(j, A)\cap subg2(i)} X_{ik}\Rightarrow Y\leftarrow(X\odot B)A^T
$$
*worse case: change edge*
$$
Y_{ij}\leftarrow \sum_{k\in N(j, A^{(i)})} X_{ik}\\
=\sum_{k}A_{jk}^{(i)}X_{ik}\\
$$
Need a three order tensor $A_{jk}^{(i)}$. Not considered now.


# Getting started

## Data structure: SparseTensor and DenseTensor

For each graph 2-Tuple Representation $X$ can have two data format. For each graph, the 2-Tuple representation is $n\times n\times d$ tensor. We have two ways to express them.
* Sparse tensor (backend.SparseTensor). indice $\in \N^{2\times nnz}$, value $\in \N ^{nnz\times d}$. 

You can create a SparseTensor ( Spare(indices, values, shape) ) as follows.

```
n, m, nnz, d = 5, 7, 17, 7
indices = torch.stack(
        (torch.randint(0, n, (nnz, )), torch.randint(0, m, (nnz, ))))
values = torch.randn((nnz, d))
A1 = torch.sparse_coo_tensor(indices, values, size=(n, m, d))
```


* Masked tensor (backend.MaskedTensor). value $\in \R^{n\times n \times d}$. mask $\{0,1\}^{n\times n}$. mask[i, j] = True means the tuple (i, j)  is not masked.

You can create a masked tensor MaskedTensor(data, mask, ) as follows.
```
B = 2
N = 3
M = 2
data = torch.randn((B, N, M))
mask = torch.zeros((B, N), dtype=torch.bool)
mask[0, :2] = True
mask[1, :1] = True
mt = MaskedTensor(data, mask, padvalue=torch.inf)
print(mt.data)
print(mt.mask)
print(mt.shape)
```

For a batch of batchsize $b$. X is
* Sparse tensor. indice $\in \N^{2\times nnz}$, value $\in \N ^{nnz\times d}$. With another batch tensor in $\N^{nnz}$
* Masked tensor. value $\in \R^{b\times n\times n \times d}$. mask $\R^{b\times n\times n}$.

For a batch. A is

* Sparse tensor of two sparse dimension when X is sparse tensor. 

* Sparse tensor of three sparse dimension when X is masked tensor.

## Basic Message passing operation.

example/nestedGNN and example/SSWL are examples for sparse and dense subgraph GNNs, respectively. 

### Sparse Representation

#### Tuple message passing
Tuple representation $X\in \R^{n\times n\times d1}$, adjacency matrix $A\in \R^{n\times n\times d2}$. d1, d2 can be the same number or any broadcastable shape.

* Message passing within subgraph, equivalent to $XA$. You can use 
```
subgnn.SpXoperator.messagepassing_tuple(A, X, "XA", datadict, aggr)
```

* Message passing across subgraph, equivalent to $AX$. You can use 
```
subgnn.SpXoperator.messagepassing_tuple(A, X, "AX", datadict, aggr)
```

* 2FWL, equivalent to $XX$. You can use 
```
subgnn.SpXoperator.messagepassing_tuple(X1, X2, "XX", datadict, aggr).
```


We also directly provide some out-of-box convolution layers in  subgnn.Spconv.

#### Tuple-wise operation

Tuple representation $X\in \R^{n\times n\times d1}$. You have a MLP $f$. To get $f(X)$. You can use
'''
X.tuplewiseapply(f)
'''

#### Pooling and Unpooling
Pooling: tuple representation to dense node representation.
```
subgnn.SpXOperator.pooling_tuple(X: SparseTensor, dim=1, pool: str = "sum")
```
Unpooling: dense node representation to tuple representation as SparseTensor. output use the same indice as tarX.
```
subgnn.SpXOperator.unpooling_node(nodeX: Tensor, tarX: SparseTensor, dim=1)
```

### Dense Representation

#### Tuple message passing
Tuple representation $X\in \R^{B\times n\times n\times d1}$, adjacency matrix $A\in \R^{B\times n\times n\times d2}$ (sparse tensor with sparse dim=3). d1, d2 can be the same number or any broadcastable shape.

* Message passing within subgraph, equivalent to $XA$. You can use 
```
subgnn.MaXoperator.messagepassing_tuple(A, X, "XA", datadict, aggr)
```

* Message passing across subgraph, equivalent to $AX$. You can use 
```
subgnn.MaXoperator.messagepassing_tuple(A, X, "AX", datadict, aggr)
```

* 2FWL, equivalent to $XX$. You can use 
```
subgnn.MaXoperator.messagepassing_tuple(X1, X2, "XX", datadict, aggr).
```


We also directly provide some out-of-box convolution layers in  subgnn.Spconv.

#### Tuple-wise operation

Tuple representation $X\in \R^{n\times n\times d1}$. You have a MLP $f$. To get $f(X)$. You can use
'''
X.tuplewiseapply(f)
'''

#### Pooling and Unpooling
Pooling: tuple representation to dense node representation.
```
subgnn.MaXOperator.pooling_tuple(X: Masked, dim=1, pool: str = "sum")
```
Unpooling: dense node representation to tuple representation as SparseTensor. output use the same indice as tarX.
```
subgnn.MaXOperator.unpooling_node(nodeX: Tensor, tarX: Masked, dim=1)
```

### Data process

We also provide utility for data processing. 

### Sparse data

When you define a pygdataset, you can change to transform input to `subgdata.SpData.sp_datapreprocess`. You can combine with a subgraph sampler as follows.

```
from functools import partial
from subgdata.SpSubgSampler import KhopSampler
kwargs={}
kwargs["transform"] = partial(sp_datapreprocess, subgsampler=partial(KhopSampler, hop=3), keys=["XA_acd", "XX_acd", "AX_acd"])
trn_d = ZINC("dataset/ZINC", **kwargs)
```
The subgraph sampler produces tuples and their structure label like shortest path distance. You can also use your own subgraph sampler policy. * Changing the `transform=...` to `pretransform=...` can save the subgraph data and thus avoids repeated data processing, however, you need to delete the preprocessed data with a different subgraph sampler.

keys means some precomputed quantities used in sparse-sparse tensor multiplication. For example, if you model use XA operation, you can add "XA_acd" to key. They will accelerate computation but take more data processing time and space. 

The transformed data have two extra properties `tupleid` of shape (2, num_tuple), `tuplefeat` of shape (2, *).

To load it, you can directly use pyg's dataloader. You can build A and X tensor as follows.

```
datadict = batch.to_dict()
A = SparseTensor(datadict["edge_index"],
                         datadict["edge_attr"],
                         shape=[datadict["num_nodes"], datadict["num_nodes"]] +
                         list(datadict["edge_attr"].shape[1:]),
                         is_coalesced=True)
X = SparseTensor(datadict["tupleid"],
                         self.tupleinit(datadict["tupleid"],
                                        datadict["tuplefeat"], datadict["x"]),
                         shape=[datadict["num_nodes"], datadict["num_nodes"]] +
                         list(datadict["edge_attr"].shape[1:]),
                         is_coalesced=True)
```


* Madata.py

Similar to sparse data, you can change to transform input to `subgdata.MaData.ma_datapreprocess`. You can combine with a subgraph sampler as follows.

```
import torch
from subgdata.MaData import ma_datapreprocess
from subgdata.MaSubgSampler import rdsampler, spdsampler
import torch_geometric.transforms as T
from functools import partial
kwargs["transform"] = partial(ma_datapreprocess, subgsampler=partial(spdsampler, hop=4))
trn_d = ZINC("dataset/ZINC", **kwargs)
```

To load it, you can still use pyg's dataloader with follow batch.

```
train_loader = DataLoader(trn_d,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  drop_last=True,
                                  num_workers=args.num_workers,
                                  follow_batch=["edge_index", "tuplefeat"])
from subgdata.MaData import batch2dense
for batch in train_loader:
        datadict = batch.to_dict()
        datadict = batch2dense(datadict)

```