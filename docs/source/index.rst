.. PyGHO documentation master file, created by
   sphinx-quickstart on Fri Sep 15 13:55:23 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

:github_url: https://github.com/GraphPKU/PygHO

PyTorch Geometric High Order Documentation
==========================================

PygHO is a library for high-order GNN. Ordinary GNNs, like GCN, GIN, GraphSage, all pass messages between nodes and produce node representations. The node representation forms a dense matrix of shape $(n, d)$, where $n$ is the number of nodes and $d$ is the hidden dimension. Existing libraries like PyG can easily implement them.

In constrast, higher-order GNNs (HOGNNs) use node tuples as the message passing unit and produce representations for the tuples. The tuple representation can be of shape $(n, n, d)$, $(n, n, n, d)$, and even more dimensions. Furthermore, to reduce complexity, the representation can be sparse. PyGHO is the first unified library for HOGNNs.

.. code-block:: latex

    >@inproceedings{PyGHO,
                    author = {Xiyuan Wang and Muhan Zhang},
                    title = {{PyGHO, a Library for High Order Graph Neural Networks}},
                    year = {2023},
    }

.. toctree::
   :glob:
   :maxdepth: 2
   :caption: Notes

   notes/installation
   notes/miniexample
   notes/datastructure
   notes/hodata
   notes/operator

.. toctree::
   :glob:
   :maxdepth: 2
   :caption: Advanced Tutorial

   notes/multtensor

.. toctree::
   :glob:
   :maxdepth: 2
   :caption: Package Reference

   modules/backend
   modules/hodata
   modules/honn
