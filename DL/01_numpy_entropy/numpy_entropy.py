#!/usr/bin/env python3
from collections import defaultdict

import numpy as np
np.seterr(divide='ignore')

if __name__ == "__main__":
    # Load data distribution, each line containing a datapoint -- a string.
    loaded_data = []
    with open("numpy_entropy_data.txt", "r") as data:
        for line in data:
            line = line.rstrip("\n")
            # TODO: Process the line, aggregating data with built-in Python
            loaded_data.append(line)

    # TODO: Create a NumPy array containing the data distribution.
    samples, data_distribution = np.unique(loaded_data, return_counts=True)
    # normalize counts
    data_distribution = data_distribution / len(loaded_data)

    # Load model distribution, each line `string \t probability`.
    model_distribution = np.zeros(data_distribution.shape)
    with open("numpy_entropy_model.txt", "r") as model:
        for line in model:
            line = line.rstrip("\n")
            # TODO: process the line, aggregating using Python data structures
            sample, dist = line.split("\t")
            if sample in samples:
                model_distribution[np.where(sample == samples)] = float(dist)

    # TODO: Compute and print the entropy H(data distribution).
    entropy = -np.sum(data_distribution * np.log(data_distribution))
    print("{:.2f}".format(entropy))

    # TODO: Compute and print cross-entropy H(data distribution, model distribution)
    cross_entropy = -np.sum(data_distribution * np.log(model_distribution))
    print("{:.2f}".format(cross_entropy))
    # and KL-divergence D_KL(data distribution, model_distribution)
    KL_divergence = np.sum(data_distribution * (np.log(data_distribution) - np.log(model_distribution)))
    print("{:.2f}".format(KL_divergence))
