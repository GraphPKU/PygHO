.. _miniexample-label:

Minimal Example
===============

Let's delve into the fundamental concepts of PyGHO using a minimal
example. The complete code can be found in
`example/minimal.py <https://github.com/GraphPKU/PygHO/tree/main/example/minimal.py>`__.
You can execute the code with the following command:

.. code:: shell

    python minimal.py

This example demonstrates the implementation of a basic HOGNN model
**Nested Graph Neural Network (NGNN)** in the paper `Nested Graph Neural
Network (NGNN) <https://arxiv.org/abs/2110.13197>`__. NGNN works by
first sampling a k-hop subgraph for each node :math:`i` and then
applying Graph Neural Networks (GNN) on all these subgraphs
simultaneously. It generates a 2-D representation
:math:`H\in \mathbb{R}^{n\times n\times d}`, where :math:`H_{ij}`
represents the representation of node :math:`j` in the subgraph rooted
at node :math:`i`. The message passing within all subgraphs can be
expressed as:

.. math::


       h_{ij}^{t+1} \leftarrow \sum_{k\in N_i(j)} \text{MLP}(h^t_{ik}),

where :math:`N_i(j)` represents the set of neighbors of node :math:`j`
in the subgraph rooted at :math:`i`. After several layers of message
passing, tuple representations :math:`H` are pooled to generate the node
representations:

.. math::


       h_i = P(\{h_{ij} | j\in V_i\}). 

This example serves as a fundamental illustration of our work.

Dataset Preprocessing
---------------------

As HOGNNs share tasks with ordinary GNNs, they can utilize datasets
provided by PyG. However, NGNN still needs to sample subgraphs,
equivalent to providing initial features for tuple representation
:math:`h_{ij}`. You can achieve this transformation with the following
code:

.. code:: python

    # Load an ordinary PyG dataset
    from torch_geometric.datasets import ZINC
    trn_dataset = ZINC("dataset/ZINC", subset=True, split="train") 

    # Transform it into a High-order graph dataset
    from pygho.hodata import Sppretransform, ParallelPreprocessDataset
    trn_dataset = ParallelPreprocessDataset(
            "dataset/ZINC_trn", trn_dataset,
            pre_transform=Sppretransform(tuplesamplers=[partial(KhopSampler, hop=3)], annotate=[""], keys=keys), num_workers=8)

The ``ParallelPreprocessDataset`` class takes a standard PyG dataset as
input and performs transformations on each graph in parallel, utilizing
8 processes in this example. The ``tuplesamplers`` parameter represents
functions that take a graph as input and produce a sparse tensor. In
this example, we use ``partial(KhopSampler, hop=3)``, a sampler designed
for NGNN, to sample a 3-hop ego-network rooted at each node. The
shortest path distance to the root node serves as the tuple features.
The produced SparseTensor is then saved and can be effectively used to
initialize tuple representations. The ``keys`` variable is a list of
strings indicating the required precomputation, which can be
automatically generated after defining a model:

.. code:: python

    from pygho.honn.SpOperator import parse_precomputekey
    keys = parse_precomputekey(model)

Mini-batch and DataLoader
-------------------------

Enabling batch training in HOGNNs requires handling graphs of varying
sizes, which can be challenging. This library concatenates the
SparseTensors of each graph along the diagonal of a larger tensor. For
instance, in a batch of :math:`B` graphs with adjacency matrices
:math:`A_i\in \mathbb{R}^{n_i\times n_i}`, node features
:math:`x\in \mathbb{R}^{n_i\times d}`, and tuple features
:math:`X\in \mathbb{R}^{n_i\times n_i\times d'}` for
:math:`i=1,2,\ldots,B`, the features for the entire batch are
represented as :math:`A\in \mathbb{R}^{n\times n}`,
:math:`x\in \mathbb{R}^{n\times d}`, and
:math:`X\in \mathbb{R}^{n\times n\times d'}`, where
:math:`n=\sum_{i=1}^B n_i`. The concatenation is as follows:

.. math::


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

We provide our DataLoader as part of PygHO. It has compatible parameters
with PyTorch's DataLoader and combines sparse tensors for different
graphs:

.. code:: python

    from pygho.subgdata import SpDataloader
    trn_dataloader = SpDataloader(trn_dataset, batch_size=128, shuffle=True, drop_last=True)

Using this DataLoader is similar to an ordinary PyG DataLoader:

.. code:: python

    for batch in dataloader:
        batch = batch.to(device, non_blocking=True)

However, in addition to PyG batch attributes (like ``edge_index``,
``x``, ``batch``), this batch also contains a SparseTensor adjacency
matrix ``A`` and initial tuple feature SparseTensor ``X``.

Learning Methods on Graphs
~~~~~~~~~~~~~~~~~~~~~~~~~~

To execute message passing on each subgraph simultaneously, you can
utilize the NGNNConv in our library:

.. code:: python

    # Definition
    self.subggnns = nn.ModuleList([
                NGNNConv(hiddim, hiddim, "sum", "SS", mlp)
                for _ in range(num_layer)
            ])

    ...
    # Forward pass
    for conv in self.subggnns:
        tX = conv.forward(A, X, datadict)
        X = X.add(tX, True)

Here, ``A`` and ``X`` are SparseTensors representing the adjacency
matrix and tuple representation, respectively. ``X.add`` implements a
residual connection.

We also provide other convolution layers, including
`GNNAK <https://arxiv.org/abs/2110.03753>`__,
`DSSGNN <https://arxiv.org/abs/2110.02910>`__,
`SSWL <https://arxiv.org/abs/2302.07090>`__,
`PPGN <https://arxiv.org/abs/1905.11136>`__,
`SUN <https://arxiv.org/abs/2206.11140>`__,
`I2GNN <https://arxiv.org/abs/2210.13978>`__, in
:py:mod:`pygho.honn.Conv`.
