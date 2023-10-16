Basic Data Structure
============================

In this section, we'll provide a refined explanation of the basic data
structures, MaskedTensor and SparseTensor, used in HOGNNs to address
their unique requirements.

MaskedTensor
------------

HOGNNs demand specialized data structures to handle high-order tensors
efficiently. One such structure is the **MaskedTensor**, consisting of
two components: ``data`` and ``mask``.

-  ``data`` has a shape of
   :math:`(\text{masked shape}, \text{dense shape})`, residing in
   :math:`\mathbb{R}^{n\times n\times d}`, where :math:`n` represents
   the number of nodes, and :math:`d` is the dimensionality of the data.
-  ``mask`` has a shape of :math:`(\text{masked shape})`, containing
   Boolean values, typically :math:`\{0,1\}^{n\times n}`. The element
   :math:`(i,j)` in ``mask`` is set to :math:`1` if the tuple
   :math:`(i,j)` exists in the tensor.

Unused elements in ``data`` do not affect the output of the operators in
this library. For example, when performing operations like summation,
MaskedTensor treats the non-existing elements as :math:`0`, effectively
ignoring them.

Here's an example of creating a MaskedTensor:

.. code:: python

    from pygho import MaskedTensor
    import torch

    n, d = 3, 3
    data = torch.tensor([[4, 1, 4], [4, 4, 2], [3, 4, 4]])
    mask = torch.tensor([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=torch.bool)
    A = MaskedTensor(data, mask)

The non-existing elements in ``data`` can be assigned arbitrary values.
The created masked tensor is as follows

.. math::


   \begin{bmatrix}
   -&1&-\\
   -&-&2\\
   3&-&-
   \end{bmatrix}

SparseTensor
------------

On the other hand, the **SparseTensor** stores only existing elements,
making it more efficient when a small ratio of valid elements is
present. A SparseTensor, with shape ``(sparse_shape, dense_shape)``,
consists of two tensors: ``indices`` and ``values``.

-  ``indices`` is an Integer Tensor with shape ``(sparse_dim, nnz)``,
   where ``sparse_dim`` represents the number of dimensions in the
   sparse shape, and ``nnz`` stands for the count of existing elements.
-  ``values`` has a shape of ``(nnz, dense_shape)``.

The columns of ``indices`` and rows of ``values`` correspond to the
non-zero elements, simplifying retrieval and manipulation of the
required information.

For instance, in the context of NGNN's representation
:math:`H\in \mathbb{R}^{n\times n\times d}`, where the total number of
nodes in subgraphs is :math:`m`, you can represent :math:`H` using
``indices`` :math:`a\in \mathbb{N}^{2\times m}` and ``values``
:math:`v\in \mathbb{R}^{m\times d}`. Specifically, for
:math:`i=1,2,\ldots,n`, :math:`H_{a_{1,i},a_{2,i}}=v_i`.

Creating a SparseTensor is illustrated in the following example:

.. code:: python

    from pygho import SparseTensor
    import torch

    n, d = 3, 3
    indices = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    values = torch.tensor([1, 2, 3])
    A = SparseTensor(indices, values, shape=(3, 3))

representing the following matrix

.. math::


   \begin{bmatrix}
   -&1&-\\
   -&-&2\\
   3&-&-
   \end{bmatrix}

Please note that in the SparseTensor format, each non-zero element is
represented by ``sparse_dim`` int64 indices and the element itself. If
the tensor is not sparse enough, SparseTensor may occupy more memory
than a dense tensor.
