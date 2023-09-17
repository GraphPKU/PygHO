## Speed issue

You can use python -O to disable all `assert` when you are sure there is no bug.

Changing the `transform` of dataset to `pre_transform` can accelerate significantly. 

Precompute spspmm's indice may provide some acceleration. (See the sparse data section)