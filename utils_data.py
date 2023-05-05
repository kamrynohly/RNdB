# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
import jax.numpy as np
from jax import random, jit, nn, vmap

import numpy as onp
from scipy.stats import norm
import math
import datasets

data_sources = {
    "simple": datasets.simple.Simple,
}

def init_D_prime(selection, n_prime, d, D=False, interval=None):
    """
    selection: text
    n_prime: int, number of samples for Dprime
    d: int, number of features in Dprime
    D: true data, only needed if near_origin is selected
    """
    if selection == "random":
        Dprime = 2 * (onp.random.random((n_prime, d)) - 0.5)
    elif selection == "rand_interval":
        a, b = interval
        Dprime = (b - a) * onp.random.random((n_prime, d)) + a
    elif selection == "near_origin":
        Dprime = D + 0.05 * onp.random.randn(n_prime, d)
    else:
        raise ValueError(
            "Supported selections are 'random', 'randomunit', and 'near_origin'"
        )
    return Dprime

def l2_loss_fn(Dprime, target_statistics, statistic_fn):
    return np.linalg.norm(statistic_fn(Dprime) - target_statistics)

@jit
def sparsemax(logits):
    """forward pass for sparsemax
    this will process a 2d-array $logits, where axis 1 (each row) is assumed to be
    the logits-vector.
    """

    # sort logits
    z_sorted = np.sort(logits, axis=1)[:, ::-1]

    # calculate k(z)
    z_cumsum = np.cumsum(z_sorted, axis=1)
    k = np.arange(1, logits.shape[1] + 1)
    z_check = 1 + k * z_sorted > z_cumsum
    k_z = logits.shape[1] - np.argmax(z_check[:, ::-1], axis=1)

    # calculate tau(logits)
    tau_sum = z_cumsum[np.arange(0, logits.shape[0]), k_z - 1]
    tau_z = ((tau_sum - 1) / k_z).reshape(-1, 1)

    return np.maximum(0, logits - tau_z)

def jit_loss_fn(statistic_fn, norm=None, lambda_l1=0):

    if norm == "L2":
        ord_norm = 2
    elif norm == "Linfty":
        ord_norm = np.inf
    else:
        ord_norm = 5

    @jit
    def compute_loss_fn(Dprime, target_statistics):
        if norm == "LogExp":
            return np.log(
                np.exp(statistic_fn(Dprime) - target_statistics).sum()
            ) + lambda_l1 * np.linalg.norm(Dprime, 1)
        else:
            return np.linalg.norm(
                statistic_fn(Dprime) - target_statistics, ord=ord_norm
            ) + lambda_l1 * np.linalg.norm(Dprime, 1)

    return compute_loss_fn

@jit
def sparsemax_project(D, feats_idx):

    return np.hstack(sparsemax(D[:, q]) for q in feats_idx)


def ohe_to_categorical(D, feats_idx):
    return np.vstack(np.argwhere(D[:, feat] == 1)[:, 1] for feat in feats_idx).T


def randomized_rounding(D, feats_idx, key, oversample=1):

    return np.hstack(
        np.vstack(
            nn.one_hot(
                random.choice(
                    key, a=len(probs), shape=(oversample, 1), p=probs
                ).squeeze(),
                len(probs),
            )
            for probs in D[:, feat]
        )
        for feat in feats_idx
    )


# Helpers from privacy part
def filter_answered_queries(
    sorted_indices: np.DeviceArray, answered_queries: np.DeviceArray
):
    return sorted_indices[
        np.in1d(sorted_indices, answered_queries,
                assume_unique=True, invert=True)
    ]

def report_worst_query(
    query_errors: np.DeviceArray,
    answered_queries: np.DeviceArray,
) -> np.DeviceArray:
    query_errors_noisy = query_errors 
    top_indices = np.flip(np.argsort(query_errors_noisy))
    return filter_answered_queries(top_indices, answered_queries)[0]

def select_noisy_q(
    query_errs: np.DeviceArray,
    answered_queries: np.DeviceArray,
    q: int,
) -> np.DeviceArray:
    q_worst_queries = np.array([])
    for i in range(q):
        answered_queries = np.append(answered_queries, q_worst_queries)
        q_worst_queries = np.append(
            q_worst_queries,
            report_worst_query(
                query_errs, answered_queries),
        )

    return np.asarray(q_worst_queries, dtype=np.int32)