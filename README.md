# PygHO

A library for high-order GNN based on torch_geometric.

## Installation
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

In constrast, higher-order GNNs use node tuples as the message passing unit and produce representations for the tuples. The tuple representation can be of shape $(n, n, d)$, $(n, n, n, d)$, and even more dimensions. Furthermore, to reduce complexity, the representation can be sparse. 

Taking NGNN (with GIN base)~\citep{NGNN} as an example, NGNN first samples a subgraph for each node $i$ and then runs GIN on all subgraphs simultaneously. It produces a 2-D representation $H\in \mathbb{R}^{n\times n\times d}$, where $H_{ij}$ represents the representation of node $j$ in the subgraph rooted at node $i$. The message passing within all subgraphs can be expressed as:

\begin{equation}
    h_{ij}^{t+1} \leftarrow \sum_{k\in N_i(j)} \text{MLP}(h^t_{ik}),
\end{equation}

where $N_i(j)$ represents the set of neighbors of node $j$ in the subgraph rooted at $i$. After several layers of message passing, tuple representations $H$ are pooled to generate the final graph representation:

\begin{equation}
    h_i = \text{P}_2\left(\big\{h_{ij} | j\in V_i\big\}\right), \quad h_{G} = \text{P}_1\left(\big\{h_i | i\in V\big\}\right),
\end{equation}

Thus, a set of operators on high-order tensors is required for HOGNN, which is the focus of our work. 


## Introduction by Example

We shortly introduce the fundamental concepts of PygHO through self-contained examples.

PygHO provides the following main features:

* Basic Data Structure

* High-Order Graph Data Preprocessing

* Mini-batches and DataLoader

* Learning Methods on Graphs

### Basic Data Structure
While basic deep learning libraries typically support the high-order tensors directly, HOGNNs demand specialized structures. NGNN, for example, employs a 2-D tensor $H\in \mathbb{R}^{n\times n\times d}$, where $H_{ij}$ represents the node representation of node $j$ in subgraph $i$. Since not all nodes are included in each subgraph, some elements in $H_{ij}$ may not correspond to any node and should not exist. To address this challenge, we introduce two distinct data structures that cater to the unique requirements of HOGNNs: MaskedTensor and SparseTensor.

#### MaskedTensor

A MaskedTensor consists of two components: `data`, with shape $(\text{masked shape}, \text{dense shape})$, and `mask`, with shape $(\text{masked shape})$. The `mask` tensor contains Boolean values, indicating whether the corresponding element in `data` exists within the tensor. For example, in the context of NGNN's representation $H\in \mathbb{R}^{n\times n\times d}$, `data` resides in $\mathbb{R}^{n\times n\times d}$, and `mask` is in $\{0,1\}^{n\times n}$. The element $(i,j)$ in `mask` is $1$ if the tuple $(i,j)$ exists in the tensor. The unused elements will not affect the output of the operators in this library. For example, the summation over a Maskedtensor will consider the non-existing elements as $0$ and thus ignore them.

For example, the following matrix 
$$
\begin{bmatrix}
0&1&0\\
0&0&2\\
3&0&0
\end{bmatrix}
$$
can be built as 
```
from pygho import MaskedTensor
n, m, nnz, d = 5, 7, 17, 7
data = torch.tensor([[4, 1, 4], [4,4,2], [3,4,4]])
mask = torch.tensor([[0, 1, 0], [0,0,1], [1,0,0]], dtype=torch.bool)
A = MaskedTensor(data, mask)
```
Here the non-existing elements in data can be set arbitrarily
#### SparseTensor

In contrast, SparseTensor stores only existing elements while ignoring non-existing ones. This approach proves to be more efficient when a small ratio of valid elements is present.  A SparseTensor, with shape (sparse\_shape, dense\_shape), comprises two tensors: `indices` (an Integer Tensor with shape (sparse\_dim, nnz)) and `values` (with shape (nnz, dense\_shape)). Here, sparse\_dim represents the number of dimensions in the sparse shape, and nnz stands for the count of existing elements. The columns of `indices` and rows of `values` correspond to the non-zero elements, making it straightforward to retrieve and manipulate the required information.

For example, in NGNN's representation $H\in \mathbb{R}^{n\times n\times d}$, assuming the total number nodes in subgraphs is $m$, $H$ can be represented as `indices` $a\in \mathbb{N}^{2\times m}$ and `values` $v\in \mathbb{R}^{m\times d}$. Specifically, for $i=1,2,\ldots,n$, $H_{a_{1,i},a_{2,i}}=v_i$.

For example, the following matrix 
$$
\begin{bmatrix}
0&1&0\\
0&0&2\\
3&0&0
\end{bmatrix}
$$
can be built as 
```
from pygho import SparseTensor
n, m, nnz, d = 5, 7, 17, 7
indices = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
values = torch.tensor([1, 2, 3])
A = SparseTensor(indices, values, shape=(3, 3))
```

Note that each non-zero elements in SpTensor format takes `sparse_dim` int64 as indices and the elements itself, if the tensor is not sparse enough, SpTensor can take more space than dense tensor. 

### High-Order Graph Data Preprocessing

HOGNNs and Message Passing Neural Networks (MPNNs) share common tasks, allowing us to reuse PyTorch Geometric's (PyG) data processing routines. However, due to the specific requirements for precomputing and preserving high-order features, we have significantly extended these routines within PyGHO. As a result, PyGHO's data processing capabilities remain highly compatible with PyG while offering convenience for HOGNNs.

#### High Order Feature Precomputation

High-order feature precomputation can be efficiently conducted in parallel using the PyGHO library. To illustrate, consider the following example:

```
# Ordinary PyG dataset
from torch_geometric.datasets import ZINC
trn_dataset = ZINC("dataset/ZINC", subset=True, split="train") 
# High-order graph dataset
from pygho.hodata import Sppretransform, ParallelPreprocessDataset
trn_dataset = ParallelPreprocessDataset(
        "dataset/ZINC_trn", trn_dataset,
        pre_transform=Sppretransform(None, tuplesamplers=[partial(KhopSampler, hop=3)], annotate=[""], keys=keys), num_workers=8)
```

The `ParallelPreprocessDataset` class takes an ordinary PyG dataset as input and performs transformations on each graph in parallel (utilizing 8 processes in this example). Here, the `tuplesamplers` parameter represents functions that take a graph as input and produce a sparse tensor. Multiple samplers can be applied simultaneously, and the resulting output is assigned the names specified in the `annotate` parameter. As an example, we use `partial(KhopSampler, hop=3)`, a sampler designed for NGNN, to sample a 3-hop ego-network rooted at each node. The shortest path distance to the root node serves as the tuple features. The produced SparseTensor is then saved and can be effectively used to initialize tuple representations.

Since the dataset preprocessing routine is closely related to data structures, we have designed two separate routines for sparse and dense tensors. These routines only differ in the `pre\_transform` function. For dense tensors, we can simply use `Mapretransform(None, tuplesamplers)`. In this case, the `tuplesamplers` is a list of functions produces a dense high-order MaskedTensor containing tuple features.

### Mini-batch and DataLoader
Enabling batch training in HOGNNs requires handling graphs of varying sizes, which is not a trivial task. Different strategies are employed for Sparse and Masked Tensor data structures.

For Sparse Tensor data, the solution is relatively straightforward. We can concatenate the tensors of each graph along the diagonal of a larger tensor: For instance, in a batch of $B$ graphs with adjacency matrices $A_i\in \mathbb{R}^{n_i\times n_i}$, node features $x\in \mathbb{R}^{n_i\times d}$, and tuple features $X\in \mathbb{R}^{n_i\times n_i\times d'}$ for $i=1,2,\ldots,B$, the features for the entire batch are represented as $A\in \mathbb{R}^{n\times n}$, $x\in \mathbb{R}^{n\times d}$, and $X\in \mathbb{R}^{n\times n\times d'}$, where $n=\sum_{i=1}^B n_i$. The concatenation is as follows,
\begin{equation}
    A=\begin{bmatrix}
        A_1&0&0&\cdots &0\\
        0&A_2&0&\cdots &0\\
        0&0&A_3&\cdots &0\\
        \vdots&\vdots&\vdots&\vdots&\vdots\\
        0&0&0&\cdots&A_B
    \end{bmatrix}
    ,x=\begin{bmatrix}
        x_1\\
        x_2\\
        x_3\\
        \vdots\\
        x_B
    \end{bmatrix}
    ,X=\begin{bmatrix}
        X_1&0&0&\cdots &0\\
        0&X_2&0&\cdots &0\\
        0&0&X_3&\cdots &0\\
        \vdots&\vdots&\vdots&\vdots&\vdots\\
        0&0&0&\cdots&X_B
    \end{bmatrix}
\end{equation} 
This arrangement allows tensors in batched data have the same number of dimension as those of a single graph and thus share common operators. We provides PygHO's own dataloader. It has the compatible parameters to PyTorch's DataLoader and further combines sparse tensors for different graphs.
```
from pygho.subgdata import SpDataloader
trn_dataloader = SpDataloader(trn_dataset, batch_size=32, shuffle=True, drop_last=True)
```

As concatenation along the diagonal leads to a lot of non-existing elements, handling Masked Tensor data involves a different strategy for saving space. In this case, tensors are padded to the same shape and stacked along a new axis. For example, in a batch of $B$ graphs with adjacency matrices $A_i\in \mathbb{R}^{n_i\times n_i}$, node features $x\in \mathbb{R}^{n_i\times d}$, and tuple features $X\in \mathbb{R}^{n_i\times n_i\times d'}$ for $i=1,2,\ldots,B$, the features for the entire batch are represented as $A\in \mathbb{R}^{B\times \tilde{n}\times \tilde{n}}$, $x\in \mathbb{R}^{B\times \tilde{n}\times d}$, and $X\in \mathbb{R}^{B\times \tilde{n}\times \tilde{n}\times d'}$, where $\tilde{n}=\max\{n_i|i=1,2,\ldots,B\}$. 
\begin{equation}
    A=\begin{bmatrix}
         \begin{pmatrix}
             A_1&0_{n_1,\tilde n-n_1}\\
             0_{\tilde n-n_1, n_1}&0_{n_1,n_1}\\
         \end{pmatrix}\\
         \begin{pmatrix}
             A_2&0_{n_2,\tilde n-n_2}\\
             0_{\tilde n-n_2, n_2}&0_{n_2,n_2}\\
         \end{pmatrix}\\
         \vdots\\
         \begin{pmatrix}
             A_B&0_{n_B,\tilde n-n_B}\\
             0_{\tilde n-n_B, n_B}&0_{n_B,n_B}\\
         \end{pmatrix}\\
    \end{bmatrix}
    ,x=\begin{bmatrix}
        \begin{pmatrix}
             x_1\\
             0_{\tilde n-n_1, d}\\
         \end{pmatrix}\\
        \begin{pmatrix}
             x_2\\
             0_{\tilde n-n_2, d}\\
         \end{pmatrix}\\
        \vdots\\
        \begin{pmatrix}
             x_B\\
             0_{\tilde n-n_B, d}\\
         \end{pmatrix}\\
    \end{bmatrix}
    ,X=\begin{bmatrix}
        \begin{pmatrix}
             X_1&0_{n_1,\tilde n-n_1}\\
             0_{\tilde n-n_1, n_1}&0_{n_1,n_1}\\
         \end{pmatrix}\\
         \begin{pmatrix}
             X_2&0_{n_2,\tilde n-n_2}\\
             0_{\tilde n-n_2, n_2}&0_{n_2,n_2}\\
         \end{pmatrix}\\
         \vdots\\
         \begin{pmatrix}
             X_B&0_{n_B,\tilde n-n_B}\\
             0_{\tilde n-n_B, n_B}&0_{n_B,n_B}\\
         \end{pmatrix}\\
    \end{bmatrix}
\end{equation}

This padding and stacking strategy ensures consistent shapes across tensors, allowing for efficient processing of dense data. We also provide the dataloader to implement it conveniently.
```
from pygho.subgdata import MaDataloader
trn_dataloader = MaDataloader(trn_dataset, batch_size=256, device=device, shuffle=True, drop_last=True)
```

### Learning Methods on Graphs

The previous section introduced novel data structures for the representation of high-order Graph Neural Networks (HOGNNs) and a novel data processing routine. Consequently, the learning methods in HOGNNs can be decomposed into operations on these tensors.

#### Code Architecture

The overall code for these operations is organized into three layers:
**Layer 1: Backend:** The `pygho.backend` layer contains basic data structures and operations on them. This layer focuses solely on tensor operations and lacks graph learning concepts. It includes:
    * Matrix multiplication: This method provides general matrix multiplication capabilities, including operations on two SparseTensors, one sparse and one MaskedTensor, and two MaskedTensors. It also supports batched matrix multiplication. Additionally, it offers operations replacing the sum in ordinary matrix multiplication with max and mean.
    * Two matrix addition: Operations for adding two sparse or two dense matrices.
    * Reduce operations: These operations include sum, mean, max, and min, which reduce dimensions in tensors.
    * Expand operation: This operation adds new dimensions to tensors.
    * Tuplewiseapply(func): It applies a given function to each element in the tensor.
    * Diagonalapply(func): This operation applies a function to diagonal elements of tensors.
**Layer 2: Graph operations:** Built upon Layer 1, the `pygho.honn.SpOperator` and `pygho.honn.MaOperator` modules provide graph operations specifically tailored for Sparse and Masked Tensor structures. Additionally, the `pygho.honn.TensorOp` layer wraps these operators, abstracting away the differences between Sparse and Masked Tensor data structures. These operations encompass:
    * General message passing between tuples: Facilitating message passing between tuples of nodes.
    * Pooling: This operation reduces high-order tensors to lower-order ones by summing, taking the maximum, or computing the mean across specific dimensions.
    * Diagonal: It reduces high-order tensors to lower-order ones by extracting diagonal elements.
    * Unpooling: This operation extends low-order tensors to high-order ones.
**Layer 3: Models:** Building on Layer 2, this layer provides a collection of representative high-order GNN layers, including NGNN, GNNAK, DSSGNN, SUN, SSWL, PPGN, and I2GNN.

Layer 3 offers numerous ready-to-use methods, and with Layer 2, users can design additional models using general graph operations. Layer 1 allows for the development of novel operations, expanding the library's flexibility and utility.

#### Usage
To illustrate how these operators work, we will use NGNN as an example. Although our operators can be applied to batched data, for simplicity, we will focus on the single-graph case. Let $H\in \mathbb{R}^{n\times n\times d}$ represent the representation matrix, and $A\in \mathbb{R}^{n\times n}$ denote the adjacency matrix. The GIN operation on all subgraphs, defined as:

\begin{equation}
    h_{ij}\leftarrow \sum_{k\in N_i(j)} \text{MLP}(h_{ik})
\end{equation}

can be represented as the following two operations:

\begin{equation}
    X' = X.\text{tuplewiseapply}(\text{MLP})
\end{equation}
This operation applies the MLP function to each tuple's representation. The matrix multiplication then sums over neighbors:
\begin{equation}
    X\leftarrow X'A^T
\end{equation}
In the matrix multiplication step, batching is applied to the last dimension of $X$. While this conversion may seem trivial, several key points are worth noting:

\begin{itemize}
    \item Optimization for induced subgraph input: In the original equation, the sum is over neighbors in the subgraph. However, the matrix multiplication version includes neighbors in the whole graph as well. Importantly, our implementation optimizes for induced subgraph cases, where neighbors outside the subgraph are automatically handled by setting their values to zero.
    \item Optimization for sparse output: The operation $X'A^T$ can produce non-zero elements for pairs $(i,j)$ that do not exist in the subgraph. For sparse input tensors $X$ and $A$, we optimize the multiplication to avoid computing such non-existent elements.
\end{itemize}
While we've illustrated the implementation of GIN as an example, our library supports the implementation of various Message Passing Neural Networks (MPNNs) on subgraphs, including GAT, GraphSage, and GCN. Message passing can also occur across subgraphs by simply transposing $X$.

Pooling processes can also be considered as a reduction of $X$. For instance:

\begin{equation}
h_i=\sum_{j\in V_i}\text{MLP}_2(h_{ij})
\end{equation}

can be implemented as follows:

\begin{python}
Xn = X.tuplewiseapply(MLP_1).sum(dim=1)
\end{python}

These examples demonstrate how our library's operators can be used to efficiently implement various MPNNs on subgraphs, providing flexibility and ease of use for HOGNNs.

## Speed issue

You can use python -O to disable all `assert` when you are sure there is no bug.

Changing the `transform` of dataset to `pre_transform` can accelerate significantly. 

Precompute spspmm's indice may provide some acceleration. (See the sparse data section)
