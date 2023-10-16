.. _hodata-label:

Efficient High Order Data Processing
====================================

In this section, we'll delve into the efficient high-order data
processing capabilities provided by PyGHO, particularly focusing on the
handling of high-order tensors, tuple feature precomputation, and data
loading strategies for both sparse and masked tensor data structures.

Adding High Order Features to PyG Dataset
-----------------------------------------

HOGNNs and MPNNs share common tasks, allowing us to leverage PyTorch
Geometric's (PyG) data processing routines. However, to cater to the
unique requirements of HOGNNs, PyGHO significantly extends these
routines while maintaining compatibility with PyG. This extension
ensures convenient high-order feature precomputation and preservation.

Efficient High-Order Feature Precomputation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

High-order feature precomputation can be efficiently conducted in
parallel using the PyGHO library. Consider the following example:

.. code:: python

    # Ordinary PyG dataset
    from torch_geometric.datasets import ZINC
    trn_dataset = ZINC("dataset/ZINC", subset=True, split="train") 
    # High-order graph dataset
    from pygho.hodata import Sppretransform, ParallelPreprocessDataset
    trn_dataset = ParallelPreprocessDataset(
            "dataset/ZINC_trn", trn_dataset,
            pre_transform=Sppretransform(tuplesamplers=partial(KhopSampler, hop=3), keys=keys), num_workers=8)

The ``ParallelPreprocessDataset`` class takes an ordinary PyG dataset as
input and performs transformations on each graph in parallel (utilizing
8 processes in this example). The ``tuplesamplers`` parameter represents
functions that take a graph as input and produce a sparse tensor. You
can apply multiple samplers simultaneously, and the resulting output can
be assigned specific names using the ``annotate`` parameter. In this
example, we utilize ``partial(KhopSampler, hop=3)``, a sampler designed
for NGNN, to sample a 3-hop ego-network rooted at each node. The
shortest path distance to the root node serves as the tuple features.
The produced SparseTensor is then saved and can be effectively used to
initialize tuple representations.

Since the dataset preprocessing routine is closely related to data
structures, we have designed two separate routines for sparse and dense
tensors. These routines only differ in the ``pre_transform`` function.
For dense tensors, we can simply use
``Mapretransform(None, tuplesamplers)``. In this case, the
``tuplesamplers`` is a function producing dense tuple features. In :py:mod:`pygho.hodata.MaTupleSampler` We provide ``spdsampler`` and
``rdsampler`` to compute shortest path distance and resisitance distance
between nodes. One example is

.. code:: python

    trn_dataset = ParallelPreprocessDataset("dataset/ZINC_trn",
                                                trn_dataset,
                                                pre_transform=Mapretransform(
                                                    partial(spdsampler,
                                                                  hop=4)),
                                                num_worker=0)

Defining Custom Tuple Samplers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In addition to the provided tuple samplers, you can define your own
tuple sampler. For sparse data, a sampler is a function or callable
object that takes a ``torch_geometric.data.Data`` object as input and
produces a sparse tensor as output. Here's an example of a custom sparse
tuple sampler that assigns ``0`` as a feature for each tuple ``(i, i)``
in the graph:

.. code:: python

    def SparseToySampler(data: PygData) -> SparseTensor:
        """
        Sample k-hop subgraph on a given PyG graph.

        Args:
        
        - data (PygData): The input PyG data.

        Returns:
        
        - SparseTensor for the precomputed tuple features.
        """
        n = data.num_nodes
        tupleid = torch.stack((torch.arange(n), torch.arange(n)))
        tuplefeat = torch.zeros((n,))
        ret = SparseTensor(tupleid, tuplefeat, shape=(n, n))
        return ret

For dense data, a sampler is a function or callable object that takes a
``torch_geometric.data.Data`` object as input and produces a tensor
along with the masked shape of the features. Here's a custom dense tuple
sampler that assigns ``0`` as a feature for each tuple ``(i, i)`` in the
graph:

.. code:: python

    def DenseToySampler(data: PygData) -> Tuple[Tensor, List[int]]:
        """
        Sample k-hop subgraph on a given PyG graph.

        Args:
        
        - data (PygData): The input PyG data.

        Returns:
        
        - Tensor: The precomputed tuple features.
        - List[int]: The masked shape of the features.
        """
        n = data.num_nodes
        val = torch.eye(n)
        return val, [n, n]

Please note that for dense data, the function returns a tuple consisting
of the value and the masked shape, as opposed to returning a
MaskedTensor. This is because the mask can typically be inferred from
the feature itself, making it unnecessary to explicitly include it in
the returned data. In such cases, the mask can be determined as val ==
1 .

Using Multiple Tuple Samplers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can use multiple tuple samplers simultaneously. For instance:

.. code:: python

    trn_dataset = ParallelPreprocessDataset(
            "dataset/ZINC_trn", trn_dataset,
            pre_transform=Sppretransform(tuplesamplers=[partial(KhopSampler, hop=1),partial(KhopSampler, hop=2)], annotate=["1hop", "2hop"], keys=keys), num_workers=8)

This code precomputes two tuple features simultaneously and assigns them
different annotations, "1hop" and "2hop," to distinguish between them.

For dense, it works similarly

.. code:: python

    trn_dataset = ParallelPreprocessDataset(
        "dataset/ZINC_trn",
        trn_dataset,
        pre_transform=Mapretransform(
            [partial(spdsampler,hop=1),partial(spdsampler,hop=2)], 
            annotate=["1hop","2hop"]),
            num_worker=0)

Sparse-Sparse Matrix Multiplication Precomputation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Efficient Sparse-Sparse Matrix Multiplication in our library can be
achieved through precomputation. The ``keys`` parameter in
``Sppretransform`` is a list of strings, where each string indicates a
specific precomputation. For example, consider the key:

::

    "X___A___1___X___0"

Here, the precomputation involves sparse matrix multiplication
:math:`AX`, but only computes the output elements that exist in
:math:`X`. These precomputation results can be shared among matrices
with the same indices. The key elements signify the following: 

- The first ``X`` refers to the target sparse matrix indices. 
- ``A`` and ``X`` represent the matrices involved in the multiplication, the adjacency matrix ``A``, and the tuple feature ``X``. 
- ``1`` denotes that dimension ``1`` of ``A`` will be reduced. 
- ``0`` signifies that dimension ``0`` of ``X`` will be reduced.

You don't need to manually feed the precomputation results to the model.
Converting the batch to a dictionary and using it as the ``datadict``
parameter is sufficient:

.. code:: python

    for batch in dataloader:
        datadict = batch.to_dict()

Dense data does not require precomputation currently.

If you use annotate in transformation, for example,

::

    Sppretransform(tuplesamplers=partial(KhopSampler, hop=1),annotate=["1hop"], keys=keys)

Then the key can be

::

    "X1hop___A___1___X1hop___0"

More details are shown in :ref:`multi-tensor-tutorial-label`
Mini-batch and DataLoader

Enabling batch training in HOGNNs demands handling graphs of varying
sizes, which presents a challenge. We employ different strategies for
Sparse and Masked Tensor data structures.

Sparse Tensor Data
~~~~~~~~~~~~~~~~~~

For Sparse Tensor data, we adopt a relatively straightforward solution.
We concatenate the tensors of each graph along the diagonal of a larger
tensor. For example, in a batch of :math:`B` graphs with adjacency
matrices :math:`A_i\in \mathbb{R}^{n_i\times n_i}`, node features
:math:`x\in \mathbb{R}^{n_i\times d}`, and tuple features
:math:`X\in \mathbb{R}^{n_i\times n_i\times d'}` for
:math:`i=1,2,\ldots,B`, the features for the entire batch are
represented as :math:`A\in \mathbb{R}^{n\times n}`,
:math:`x\in \mathbb{R}^{n\times d}`, and
:math:`X\in \mathbb{R}^{n\times n\times d'}`, where
:math:`n=\sum_{i=1}^B n_i`. This arrangement allows tensors in batched

data to have the same number of dimensions as those of a single graph,
facilitating the sharing of common operators.

We provide PygHO's own DataLoader to simplify this process:

.. code:: python

    from pygho.subgdata import SpDataloader
    trn_dataloader = SpDataloader(trn_dataset, batch_size=32, shuffle=True, drop_last=True)

Masked Tensor Data
~~~~~~~~~~~~~~~~~~

As concatenation along the diagonal leads to a lot of non-existing
elements, handling Masked Tensor data involves a different strategy for
saving space. In this case, tensors are padded to the same shape and
stacked along a new axis. For instance, in a batch of :math:`B` graphs
with adjacency matrices :math:`A_i\in \mathbb{R}^{n_i\times n_i}`, node
features :math:`x\in \mathbb{R}^{n_i\times d}`, and tuple features
:math:`X\in \mathbb{R}^{n_i\times n_i\times d'}` for
:math:`i=1,2,\ldots,B`, the features for the entire batch are
represented as
:math:`A\in \mathbb{R}^{B\times \tilde{n}\times \tilde{n}}`,
:math:`x\in \mathbb{R}^{B\times \tilde{n}\times d}`, and
:math:`X\in \mathbb{R}^{B\times \tilde{n}\times \tilde{n}\times d'}`,
where :math:`\tilde{n}=\max\{n_i|i=1,2,\ldots,B\}`.

.. math::


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

The 0 for padding will be masked in the result MaskedTensor.

We also provide a DataLoader for this purpose:

.. code:: python

    from pygho.subgdata import MaDataloader
    trn_dataloader = MaDataloader(trn_dataset, batch_size=256, device=device, shuffle=True, drop_last=True)

This padding and stacking strategy ensures consistent shapes across
tensors, allowing for efficient processing of dense data.

