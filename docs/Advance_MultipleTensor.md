# Multiple Tensor

In our dataset preprocessing routine, the default computation involves two high-order tensors: the adjacency matrix `A` and the tuple feature `X`. However, in certain scenarios, there may be a need for additional high-order tensors. For instance, when using a Nested Graph Neural Network (GNN) with a 2-hop GNN as the base GNN, Message Passing Neural Network (MPNN) operations are performed on each subgraph with an augmented adjacency matrix. In this case, two high-order tensors are required: the tuple feature and the augmented adjacency matrix.

During data preprocessing, we can use multiple samplers, each responsible for sampling one tensor. For sparse data, the code might look like this:

```python
trn_dataset = ParallelPreprocessDataset(
    "dataset/ZINC_trn", trn_dataset,
    pre_transform=Sppretransform(
        tuplesamplers=[
            partial(KhopSampler, hop=3),
            partial(KhopSampler, hop=2)
        ],
        annotate=["tuplefeat", "2hopadj"],
        keys=keys
    ),
    num_workers=8
)
```

In this code, two tuple features are precomputed simultaneously and assigned different annotations: "tuplefeat" and "2hopadj" to distinguish between them.

For dense data, the process is quite similar:

```python
trn_dataset = ParallelPreprocessDataset(
    "dataset/ZINC_trn",
    trn_dataset,
    pre_transform=Mapretransform(
        [
            partial(spdsampler, hop=3),
            partial(spdsampler, hop=2)
        ], 
        annotate=["tuplefeat", "2hopadj"]
    ),
    num_worker=0
)
```

After passing the data through a dataloader, the batch will contain `Xtuplefeat` and `X2hopadj` as the high-order tensors that are needed. For dense models, this concludes the process. However, for sparse models, if you want to retrieve the correct keys, you will need to modify the operator symbols for sparse message passing layers.

Ordinarily, the `NGNNConv` is defined as:

```python
NGNNConv(hiddim, hiddim, mlp=mlp)
```

This is equivalent to:

```python
NGNNConv(hiddim, hiddim, mlp=mlp, optuplefeat="X", opadj="A")
```

To ensure that you retrieve the correct keys, you should use:

```python
NGNNConv(hiddim, hiddim, mlp=mlp, optuplefeat="Xtuplefeat", opadj="X2hopadj")
```

Similar modifications should be made for other layers as needed.