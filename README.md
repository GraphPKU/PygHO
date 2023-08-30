TODO: fix spmamm and mamamm (for min, max, min aggregation)
# PygHO (in progress)

A library for high-order GNN based on torch_geometric.

## Installation
We have tested `PygHO` in the following environments
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
First clone our repo
```
git clone https://github.com/GraphPKU/PygHO.git
```
Then install it locally
```
cd PygHO
pip install -e ./
```
`-e` enables modifying the library code dynamically and is optional. 


## Preliminary

PygHO is a library for high-order GNN. Ordinary GNNs, like GCN, GIN, GraphSage, all pass messages between nodes and produce node representations. The node representation forms a dense matrix of shape $(n, d)$, where $n$ is the number of nodes and $d$ is the hidden dimension. Existing libraries like PyG can easily implement them.

In constrast, higher-order GNNs use node tuples as the message passing unit and produce representations for the tuples. The tuple representation can be of shape $(n, n, d)$, $(n, n, n, d)$, and even more dimensions. Furthermore, to reduce complexity, the representation can be sparse. Our library aims to provide support for them. 

## Introduction by Example

We shortly introduce the fundamental concepts of PygHO through self-contained examples.

PygHO provides the following main features:

* Basic Data Structure

* High-Order Graph Data Preprocessing

* Mini-batches and DataLoader

* Learning Methods on Graphs

### Basic Data Structure
We provide data structures for high order tensors: Sparse and Dense. In implementation, they are very different. However, they are essentially represent high-order tensors.

#### Sparse Data Structure
The ordinary tensor used in machine learning stores each element contiguously in physical memory. However, in graph machine learning, some tensors (like graph adjacency matrix) have almost all its elements equals to $0$ (in other words, have few non-zero elements and are sparse). Therefore, sparse tensor can be applied for saving memory and time. 

Various sparse storage formats have been developed over the years. Among them, PygHO implements COO format for sparse tensor. The non-zero elements are stored as tuples of element indices and the corresponding values.

* The indices of non-zero elements are collected in a `indices` LongTensor of shape `(sparsedim, nnz)`, where `sparsedim` the number of sparse dimensions and `nnz` is the number of non-zero elements.

* The corresponding element values are collected in `values` tensor of shape `(nnz, denseshape)` of the value type, where `denseshape` can be arbitrary.

For example, the following matrix 
$$
\begin{bmatrix}
0&1&0\\
0&0&2\\
3&0&0
\end{bmatrix}
$$
can be represented as 
```
indices = [[0, 1, 2], [1, 2, 0]]
values = [1, 2, 3]
```

To create a SparseTensor, we can use `SpTensor` class.
```
from pygho import SparseTensor
n, m, nnz, d = 5, 7, 17, 7
indices = torch.stack(
        (torch.randint(0, n, (nnz, )), torch.randint(0, m, (nnz, ))))
values = torch.randn((nnz, d))
A = SparseTensor(indices, values, shape=(n, m, d))
```

Note that each non-zero elements in SpTensor format takes `sparse_dim` int64 as indices and the elements itself, if the tensor is not sparse enough, SpTensor can take more space than dense tensor. 

#### Dense Data Structure

For tensors not sparse enough, we still need to use dense tensor to store data but keeps sparsity: ignore elements that should not exists. For example, as graph size varies, the size of embedding changes and padding is needed. Therefore, we propose MaskedTensor: a dense tensor `data` to save values of the tensor and a BoolTensor `mask` to show whether an elements exists in the tensor.

You can create a masked tensor MaskedTensor(data, mask) as follows.
```
B = 2
N = 3
M = 2
data = torch.randn((B, N, M))
mask = torch.zeros((B, N), dtype=torch.bool)
mask[0, :2] = True
mask[1, :1] = True
mt = MaskedTensor(data, mask, padvalue=torch.inf)
```
where `mask`'s shape should match to first dimensions of `data`. Elements in mask is true iff the corresponding elements in data exists in the tensor. 

MaskedTensor is useful when the tensor is not sparse.

#### Basic Operations
Both MaskedTensor and SparseTensor have the following operations.
* to(device)

Transfer all data to device.

* shape

Return the shape of tensor

* reduce operation: sum, mean, max, min

Reduce dimensions.

* tuplewiseapply(func)

Apply function to each elements.

### High-Order Graph Data Preprocessing

In general, we wrap Pyg's dataset preprocessing routine.

SubgDatasetClass is a wrapper that changes the preprocessed name with different pre_transform function. Otherwise, you need to delete the preprocessed dataset with different data pre_transform functions.

```
from pygho.subgdata import SubgDatasetClass
trn_dataset = SubgDatasetClass(ZINC)("dataset/ZINC",
                   subset=True,
                   split="train")
```
To get high-order information, we wrap pre_transform function.

#### Sparse

For sparse, we can simply use `Sppretransform(your_tuple_sampler)` as the pre_transform function. 
```
from pygho.subgdata import Sppretransform, SubgDatasetClass
from pygho.subgdata.SpSubgSampler import KhopSampler
trn_dataset = SubgDatasetClass(ZINC)("dataset/ZINC",
                   subset=True,
                   split="train",
                   pre_transform=Sppretransform(partial(KhopSampler, hop=3)))
```

After preprocess, each data contains all keys in original data, `tupleid` (the indices of tuples), and `tuplefeat` (extra features of tuples assigned by tuple sampler).

#### Dense

For dense, we can simply use `Mapretransform(your_tuple_sampler)` as the pre_transform function.

```
from pygho.subgdata import SubgDatasetClass, Mapretransform
from pygho.subgdata.MaSubgSampler import spdsampler

trn_dataset = SubgDatasetClass(ZINC)("dataset/ZINC",
                   subset=True,
                   split="train",
                   pre_transform=Mapretransform(partial(spdsampler, hop=4)))
```

After preprocess, each data contains all keys in original data, `tuplemask` (the mask of tuples), and `tuplefeat` (extra features of tuples assigned by tuple sampler).
### Mini-batch and DataLoader

We provides dataloader for sparse and dense data respectively.

```
from pygho.subgdata import SpDataloader
trn_dataloader = SpDataloader(trn_dataset, batch_size=32, shuffle=True, drop_last=True)
```
For a batch of size $b$, `tupleid` $\in \mathbb{N}^{2\times nnz}$, `tuplefeat` $\in \mathbb{N} ^{nnz\times d}$, where `nnz` is sum of the number of non-zero elements in each sparse data. The initial node representation can be a sparse tensor of shape $(N, N, *)$.
```
X = SparseTensor(batch.tupleid,
                batch.tuplefeat,
                shape=[batch.num_nodes, batch.num_nodes] + list(batch.tuplefeat.shape[1:]))
```
$X$ is a block diagonal matrix, whose blocks correspond to initial representations of each data.

The adjacency is also a block diagonal matrix, whose blocks correspond to the adjacency matrix of each data.


For dense, 
```
from pygho.subgdata import MaDataloader
trn_dataloader = MaDataloader(trn_dataset, batch_size=256, device=device, shuffle=True, drop_last=True)
```

For a batch of size $b$. 
`tuplefeat` is of shape $(b, n, n, *)$. `tuplemask` is of shape $(b, n, n)$. `A` is sparse adjacency matrix of shape (b, n, n, *). 

### Learning Methods on Graph
example/nestedGNN and example/SSWL are examples of sparse and dense subgraph GNNs, respectively. 
#### Basic Message passing operation.
Tuple representation $X\in \mathbb{R}^{n\times n\times d1}$, adjacency matrix $A\in \mathbb{R}^{n\times n\times d2}$. d1, d2 can be the same number or any broadcastable shape.

* Message passing within subgraph, equivalent to $XA$. You can use 
```
subgnn.SpXoperator.messagepassing_tuple(X, A, "X_1_A_0", datadict, aggr)
```

* Message passing across subgraph, equivalent to $AX$. You can use 
```
subgnn.SpXoperator.messagepassing_tuple(A, X, "A_1_X_0", datadict, aggr)
```

* 2FWL, equivalent to $XX$. You can use 
```
subgnn.SpXoperator.messagepassing_tuple(X1, X2, "X_1_X_0", datadict, aggr).
```

We also directly provide some out-of-box convolution layers in subgnn.Spconv.

#### Pooling and Unpooling
Pooling: tuple representation to dense node representation.
```
subgnn.SpXOperator.pooling2nodes(X: SparseTensor, dim=1, pool: str = "sum")
subgnn.SpXOperator.pooling2tuples(X: SparseTensor, dim=1, pool: str = "sum")
```
Unpooling: 
```
subgnn.SpXOperator.unpooling4nodes(nodeX: Tensor, tarX: SparseTensor, dim=1)
subgnn.SpXOperator.unpooling4tuples(srcX: SparseTensor, tarX: SparseTensor, dim=1)
```

### Dense Representation

#### Tuple message passing
Tuple representation $X\in \mathbb{R}^{B\times n\times n\times d1}$, adjacency matrix $A\in \mathbb{R}^{B\times n\times n\times d2}$ (sparse tensor with sparse dim=3). d1, d2 can be the same number or any broadcastable shape.

* Message passing within subgraph, equivalent to $XA$. You can use 
```
subgnn.MaXoperator.messagepassing_tuple(X, A, datadict, aggr)
```

* Message passing across subgraph, equivalent to $AX$. You can use 
```
subgnn.MaXoperator.messagepassing_tuple(A, X, datadict, aggr)
```

* 2FWL, equivalent to $XX$. You can use 
```
subgnn.MaXoperator.messagepassing_tuple(X1, X2, datadict, aggr).
```


We also directly provide some out-of-box convolution layers in subgnn.Spconv.


#### Pooling and Unpooling
Pooling: tuple representation to dense node representation.
```
subgnn.MaXOperator.pooling_tuple(X: Masked, dim=1, pool: str = "sum")
```
Unpooling: dense node representation to tuple representation as MaskedTensor. Output uses the same mask as tarX.
```
subgnn.MaXOperator.unpooling_node(nodeX: Tensor, tarX: Masked, dim=1)
```

## Speed issue

You can use python -O to disable all `assert` when you are sure there is no bug.

Changing the `transform` of dataset to `pre_transform` can accelerate significantly. 

Precompute spspmm's indice may provide some acceleration. (See the sparse data section)
