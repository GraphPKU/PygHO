# Operators
In this section, we'll provide a detailed introduction of some basic operators on high order tensors.
## Code Architecture
The code for these high-order graph neural network (HOGNN) operations is organized into three layers:

**Layer 1: Backend:** This layer, found in the `pygho.backend` module, contains basic data structures and operations focused on tensor manipulations. It lacks graph-specific learning concepts and includes the following functionalities:

- **Matrix multiplication:** This method supports general matrix multiplication capabilities, including operations on two SparseTensors, one sparse and one MaskedTensor, and two MaskedTensors. It also handles batched matrix multiplication and offers operations that replace the sum in traditional matrix multiplication with max and mean operations.
- **Two matrix addition:** Operations for adding two sparse or two dense matrices.
- **Reduce operations:** These operations include sum, mean, max, and min, which reduce dimensions in tensors.
- **Expand operation:** This operation adds new dimensions to tensors.
- **Tuplewise apply(func):** It applies a given function to the underlying data tensor.
- **Diagonal apply(func):** This operation applies a function to diagonal elements of tensors.

**Layer 2: Graph operations:** Building upon Layer 1, the `pygho.honn.SpOperator` and `pygho.honn.MaOperator` modules provide graph operations specifically tailored for Sparse and Masked Tensor structures. Additionally, the `pygho.honn.TensorOp` layer wraps these operators, abstracting away the differences between Sparse and Masked Tensor data structures. These operations encompass:

- **General message passing between tuples:** Facilitating message passing between tuples of nodes.
- **Pooling:** This operation reduces high-order tensors to lower-order ones by summing, taking the maximum, or computing the mean across specific dimensions.
- **Diagonal:** It reduces high-order tensors to lower-order ones by extracting diagonal elements.
- **Unpooling:** This operation extends low-order tensors to high-order ones.

**Layer 3: Models:** Building on Layer 2, this layer provides a collection of representative high-order GNN layers, including NGNN, GNNAK, DSSGNN, SUN, SSWL, PPGN, and I2GNN. Layer 3 offers numerous ready-to-use methods, and with Layer 2, users can design additional models using general graph operations. Layer 1 allows for the development of novel operations, expanding the library's flexibility and utility. Now let's explore these layers in more detail.

### Layer 1: Backend

#### Spspmm
One of the most complex operators in this layer is sparse-sparse matrix multiplication (Spspmm). Given two sparse matrices, C and D, and assuming their output is B:

$$
B_{ij} = \sum_k C_{ik} D_{kj}
$$

The Spspmm operator utilizes a coo format to represent elements. Assuming element C_{ik} corresponds to C.values[c_{ik}] and D_{kj} corresponds to D.values[d_{kj}], we can create a tensor `bcd` of shape (3, m), where m is the number of pairs (i, j, k) where both C_{ik} and D_{kj} exist. The multiplication can be performed as follows:

```python
B.values = zeros(...)
for i in range(m):
    B.values[bcd[0, i]] += C.values[bcd[1, i]] * D.values[bcd[2, i]]
```

This summation process can be efficiently implemented in parallel on GPU using `torch.Tensor.scatter_reduce_`. The `bcd` tensor can be precomputed with `pygho.backend.Spspmm.spspmm_ind` and shared among matrices with the same indices.

Hadamard product between two sparse matrices can also be implemented, where $C = A \odot B$: C can use the same indice as $A$.

```python
C.values[a_{ij}] = A.values[a_{ij}] * B.values[b_{ij}]
```

The tensor `b2a` can be defined, where `b2a[b_ij] = a_ij` if A has the element (i, j); otherwise, it is set to -1. Then, the Hadamard product can be computed as follows:

```python
C.values = zeros(...)
for i in range(A.nnz):
    if b2a[i] >= 0:
        C.values[i] = A.values[i] * B.values[b2a[i]]
```

The operation can also be efficiently implemented in parallel on a GPU.

To compute $A\odot (CD)$, you can define a tensor `acd` of shape (3, m') where `acd[0] = b2a[bcd[0]]`, `acd[1] = bcd[1]`, and `acd[2] = bcd[2]`, and remove columns i where `acd[0, i] = -1`. The computation can be done as follows:

```python
ret.values = zeros(...)
for i in range(acd.shape[1]):
    ret.values[acd[0, i]] += A.values[acd[0, i]] * B.values[acd[1, i]] * C.values[acd[2, i]]
```

Like the previous operations, this can also be implemented efficiently in parallel on a GPU. Additionally, by setting `A.values[acd[0, i]]` to 1, A can act as a mask, ensuring that only elements existing in A are computed.

The overall wrapper for these functions is `pygho.honn.Spspmm.spspmm`, which can perform sparse-sparse matrix multiplication with precomputed indices. `pygho.honn.Spspmm.spspmpnn` provides a more complex operator that goes beyond matrix multiplication, allowing you to implement various graph operations. It can in fact implement the following framework.

$$
ret_{ij} = \phi(\{(A_{ij}, B_{ik}, C_{kj})|B_{ik},C_{kj} \text{ elements exist}\})
$$
where `phi` is a general multiset function, which is a functional parameter of `spspmpnn`. With it, we can implement GAT on each subgraph ...??.

#### TuplewiseApply

Both Sparse and Masked Tensors have the `tuplewiseapply` function. The most common usage is:

```python
mlp = ...
X.tuplewiseapply(mlp)
```

However, in practice, this function directly applies the values or data tensor to `mlp`. As linear layers, non-linearities, and layer normalization all operate on the last dimension, this operation is essentially equivalent to tuplewise apply. For batchnorm, we provide a version that not affected by this problem in `pygho.honn.utils`.

#### DiagonalApply

Both Sparse and Masked Tensors have the `diagonalapply` function. Unlike `tuplewiseapply`, this function passes both data/values and a mask indicating whether the corresponding elements are on the diagonal to the input function. A common use case is:

```python
mlp1 = ...
mlp2 = ...
lambda x, diagonalmask: torch.where(diagonalmask, mlp1(x), mlp2(x))
X.diagonalapply(mlp)
```

Here, `mlp1` is applied to diagonal elements, and `mlp2` is applied to non-diagonal elements. You can also use `torch_geometric.nn.HeteroLinear` for a faster implementation.

### Layer 2: Operators

`pygho.honn.SpOperator` and `pygho.honn.MaOperator` wrap the backend for SparseTensor and MaskedTensor separately. Their APIs are unified in `pygho.honn.TensorOp`. The basic operators include `OpNodeMessagePassing` (node-level message passing), `OpMessagePassing` (tuple-level message passing, wrapping matrix multiplication), `OpPooling` (reduce high-order tensors to lower-order ones by sum, mean, max), `OpDiag` (reduce high-order tensors to lower-order ones by extracting diagonal elements), and `OpUnpooling` (extend lower-order tensors to higher-order ones). Special cases are also defined.

#### Sparse OpMessagePassing

As described in Layer 1, the `OpMessagePassing` operator wraps the properties of Spspmm and is defined with parameters like `op0`, `op1`, `dim1`, `op2`, `dim2`, and `aggr`. It retrieves precomputed data from a data dictionary during the forward process using keys like `f"{op0}___{op1}___{dim1}___{op2}___{dim2}"`. Here's the forward method signature:

```python
def forward(self,
            A: SparseTensor,
            B: SparseTensor,
            datadict: Dict,
            tarX: Optional[SparseTensor] = None) -> SparseTensor:
```

In this signature, `tarX` corresponds to `op0`, providing the target indices, while `A` and `B` correspond to `op1` and `op2`. The `datadict` can be obtained from the data loader using `for batch in dataloader: batch.to_dict()`.

## Example

To illustrate how these operators work, let's use NGNN as an example. Although our operators can handle batched data, for simplicity, we'll focus on the single-graph case. Let H represent the representation matrix in $\mathbb{R}^{n\times n\times d}$, and A denote the adjacency matrix in $\mathbb{R}^{n\times n}$. The Graph Isomorphism Network (GIN) operation on all subgraphs can be defined as:

$$
h_{ij} \leftarrow \sum_{k\in N_i(j)} \text{MLP}(h_{ik})
$$

This operation can be represented using two steps:

1. Apply the MLP function to each tuple's representation:

```python
X' = X.tuplewiseapply(MLP)
```

2. Perform matrix multiplication to sum over neighbors:

```python
X = X' * A^T
```

In the matrix multiplication step, batching is applied to the last dimension of X. This conversion may seem straightforward, but there are several key points to consider:

- Optimization for induced subgraph input: The original equation involves a sum over neighbors in the subgraph, but the matrix multiplication version includes neighbors from the entire graph. However, our implementation optimizes for induced subgraph cases, where neighbors outside the subgraph are automatically handled by setting their values to zero.

- Optimization for sparse output: The operation X' * A^T may produce non-zero elements for pairs (i, j) that do not exist in the subgraph. For sparse input tensors X and A, we optimize the multiplication to avoid computing such non-existent elements.

Pooling processes can also be considered as a reduction of $X$. For instance:

$$
h_i=\sum_{j\in V_i}\text{MLP}_2(h_{ij})
$$

can be implemented as follows:

```
# definition
self.pool = OpPoolingSubg2D(...)
...
# forward
Xn = self.pool(X.tuplewiseapply(MLP_1))
```

This example demonstrate how our library's operators can be used to efficiently implement various HOGNNs.