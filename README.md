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


# Implementation

## 2-Tuple Representation $X$

* For each graph, the 2-Tuple representation is $n\times n\times d$ tensor. We have two ways to express them.
  * Sparse tensor (backend.SparseTensor). indice $\in \N^{2\times nnz}$, value $\in \N ^{nnz\times d}$. 
  * Masked tensor (backend.MaskedTensor). value $\in \R^{n\times n \times d}$. mask $\R^{n\times n}$.
* For a batch of batchsize $b$. X is
  * Sparse tensor (backend.SparseTensor). indice $\in \N^{2\times nnz}$, value $\in \N ^{nnz\times d}$. With another batch tensor in $\N^{nnz}$
  * Masked tensor  (backend.MaskedTensor). value $\in \R^{b\times n\times n \times d}$. mask $\R^{b\times n\times n}$.

## Adjacency Matrix

For a batch. A is

* Sparse tensor of two sparse dimension when X is sparse tensor. 
* Sparse tensor of three sparse dimension when X is masked tensor.

# Data

* Spdata.py

Sparse data

contain tupleid (node tuples, (2, num_tuples)), tuplelabel (node tuple features, (num_tuples, d)), the precomputed spspmm indices, 

SubgSampler class: produce subgid, subglabel (need a lot of policies)

DataPreprocessor: use SubgSampler to produce subgid, subglabel. Optional : spspmm's precomputation. (TODO: lazy precompute/cache)

* Madata.py

Todo.

contains 1-d X, 

pad and mask in collate function
