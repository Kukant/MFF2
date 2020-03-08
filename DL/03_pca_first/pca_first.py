#!/usr/bin/env python3
import argparse
import os
import sys
import time

import numpy as np
import tensorflow as tf

from mnist import MNIST

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples", default=256, type=int, help="MNIST examples to use.")
    parser.add_argument("--iterations", default=100, type=int, help="Iterations of the power algorithm.")
    parser.add_argument("--seed", default=42, type=int, help="Random seed.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--verbose", default=False, action="store_true", help="Verbose TF logging.")
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Fix random seeds and threads
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Report only errors by default
    if not args.verbose:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    # Load data
    mnist = MNIST()

    data = tf.convert_to_tensor(mnist.train.data["images"][:args.examples])
    # TODO: Data has shape [args.examples, MNIST.H, MNIST.W, MNIST.C].
    # We want to reshape it to [args.examples, MNIST.H * MNIST.W * MNIST.C].
    # We can do so using `tf.reshape(data, new_shape)` with new shape
    # `[data.shape[0], data.shape[1] * data.shape[2] * data.shape[3]]`.
    data = tf.reshape(data, (data.shape[0], data.shape[1] * data.shape[2] * data.shape[3]))

    # TODO: Now compute mean of every feature. Use `tf.math.reduce_mean`,
    # and set `axis` to zero -- therefore, the mean will be computed
    # across the first dimension, so across examples.
    mean = tf.math.reduce_mean(data, axis=0)

    # TODO: Compute the covariance matrix. The covariance matrix is
    #   (data - mean)^T * (data - mean) / data.shape[0]
    # where transpose can be computed using `tf.transpose` and matrix
    # multiplication using either Python operator @ or `tf.linalg.matmul`.
    cov = tf.transpose(data - mean) @ (data - mean) / data.shape[0]

    # TODO: Compute the total variance, which is sum of the diagonal
    # of the covariance matrix. To extract the diagonal use `tf.linalg.diag_part`
    # and to sum a tensor use `tf.math.reduce_sum`.
    total_variance = tf.math.reduce_sum(tf.linalg.diag_part(cov))

    # TODO: Now run `args.iterations` of the power iteration algorithm.
    # Start with a vector of `cov.shape[0]` ones of type tf.float32 using `tf.ones`.
    v = tf.ones(cov.shape[0], dtype=tf.float32)
    for i in range(args.iterations):
        # TODO: In the power iteration algorithm, we compute
        # 1. v = cov * v
        #    The matrix-vector multiplication can be computed using `tf.linalg.matvec`.
        v = tf.linalg.matvec(cov, v)
        # 2. s = l2_norm(v)
        #    The l2_norm can be computed using `tf.linalg.norm`.
        s = tf.linalg.norm(v)
        # 3. v = v / s
        v /= s

    # The `v` is now the eigenvector of the largest eigenvalue, `s`. We now
    # compute the explained variance, which is a ration of `s` and `total_variance`.
    explained_variance = s / total_variance

    # TODO: Write the total variance and explained variance in percents rounded to two decimal places.
    with open("pca_first.out", "w") as out_file:
        print("{:.2f} {:.2f}".format(total_variance, 100 * explained_variance), file=out_file)
