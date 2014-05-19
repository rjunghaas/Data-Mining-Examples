Data Mining Examples README

This is some sample code from assignments in a Data Mining course from UC Berkeley's
School of Information (INFO 290T).  The code presented here is for:

1. Calculating the Gini Index of a split.  This is used to calculate the optimal partitions
when constructing a decision tree.  Typically, the algorithm will iterate over all splits,
calculate the Gini, and keep the split with the minimum score.

2. Back Propagation.  This is the algorithm for constructing a neural network.  The
algorithm will converge upon the optimal weights for each node which can then offer the
best predicted output value.  This is an example of basic machine learning.

3. K-Means.  This is code to demonstrate how to do simple clustering on a single dimension.
The algorithm clusters and then checks for the minimum distance before re-optimizing until
no further changes are necessary.