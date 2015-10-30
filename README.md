# cp-lite
A simple bare-bones implementation of a conformal predictor for use in a "batch" setting where there is a fixed set of training data (the "bag" in CP parlance) and we wish to compute p-values for a number of held-out test examples.  This setting does *not* match the fully online framework for which CPs were designed; hence, there are no theoretical guarantees (e.g. CP "validity") here.

This implementation is not suitable for large-scale problems with many examples and/or high dimensions.  In these cases, a more efficient way of computing nearest neighbors should be employed (e.g. Locality-sensitive hashing perhaps).  Also, the expensive **sort()** calls in **knn_cp.m** could be replaced by a more efficient selection algorithm.

### Quick Start

Get a copy of the USPS data set:
```
    cd Test
    ./getUSPS.sh
```
Then, from within Matlab, change to the "Test" directory and do
```
    test_conformal_flowers
    test_usps
```

### References
o  Schafer, Vovk "A tutorial on conformal prediction," [arXiv](http://arxiv.org/abs/0706.3188)
