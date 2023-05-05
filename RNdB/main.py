# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
import logging
import os

import configargparse
import pandas as pd
from jax import numpy as np, random

from relaxed_adaptive_projection import RAPConfiguration, RAP
from relaxed_adaptive_projection.constants import Norm
from utils_data import data_sources, ohe_to_categorical

parser = configargparse.ArgumentParser()
parser.add_argument(
    "--config-file",
    "-c",
    required=False,
    is_config_file=True,
    help="Path to config file",
)
parser.add_argument(
    "--num-dimensions",
    "-d",
    type=int,
    default=2,
    dest="d",
    help="Number of dimensions in the "
    "original dataset. Does not need to "
    "be set when consuming csv files ("
    "default: 2)",
)
parser.add_argument(
    "--num-points",
    "-n",
    type=int,
    default=1000,
    dest="n",
    help="Number of points in the original "
    "dataset. Only used when generating "
    "datasets (default: 1000)",
)

parser.add_argument(
    "--num-generated-points",
    "-N",
    type=int,
    default=1000,
    dest="n_prime",
    help="Number of points to " "generate (default: " "1000)",
)
parser.add_argument(
    "--iterations", type=int, default=1000, help="Number of iterations (default: 1000)"
)
parser.add_argument(
    "--data-source",
    type=str,
    choices=data_sources.keys(),
    default="toy_binary",
    dest="data_source",
    help="Data source used to train data generator",
)
parser.add_argument(
    "--read-file",
    type=bool,
    default=False,
    help="Choose whether to regenerate or read data from file "
    "for randomly generated datasets",
)

parser.add_argument(
    "--use-data-subset",
    type=bool,
    default=False,
    dest="use_subset",
    help="Use only n rows and d "
    "columns of the data read "
    "from the file as input to "
    "the algorithm. Will not "
    "affect random inputs.",
)
parser.add_argument("--filepath", type=str, default="", help="File to read from")
parser.add_argument(
    "--seed", type=int, default=0, help="Seed to use for random number generation"
)
parser.add_argument(
    "--statistic-module",
    type=str,
    default="statistickway",
    help="Module containing preserve_statistic "
    "function that defines statistic to be "
    "preserved. Function MUST be named "
    "preserve_statistic",
)
parser.add_argument("--k", type=int, default=3, help="k-th marginal (default k=3)")
parser.add_argument(
    "--workload", type=int, default=64, help="workload of marginals (default 64)"
)
parser.add_argument(
    "--learning-rate",
    "-lr",
    type=float,
    default=1e-3,
    help="Adam learning rate (default: 1e-3)",
)
parser.add_argument(
    "--initialize_binomial",
    type=bool,
    default=False,
    help="Initialize with 1-way marginals",
)
parser.add_argument(
    "--lambda-l1", type=float, default=0, help="L1 regularization term (default: 0)"
)
parser.add_argument(
    "--stopping-condition",
    type=float,
    default=10 ** -7,
    help="If improvement on loss function is less than stopping condition, RAP will be terminated",
)
parser.add_argument(
    "--all-queries",
    action="store_true",
    help="Choose all q queries, no selection step. WARNING: this option overrides the top-q argument",
)
parser.add_argument(
    "--top-q", type=int, default=50, help="Top q queries to select (default q=500)"
)
parser.add_argument(
    "--epochs", type=int, default=10, help="Number of epochs (default: 100)"
)
parser.add_argument(
    "--csv-path",
    type=str,
    default="results",
    dest="csv_path",
    help="Location to save results in csv format",
)
parser.add_argument("--silent", "-s", action="store_true", help="Run silently")
parser.add_argument("--verbose", "-v", action="store_true", help="Run verbose")
parser.add_argument(
    "--norm",
    type=str,
    choices=["Linfty", "L2", "L5", "LogExp"],
    default="L2",
    help="Norm to minimize if using the optimization paradigm (default: L2)",
)

parser.add_argument(
    "--categorical-consistency",
    action="store_true",
    help="Enforce consistency categorical variables",
)
parser.add_argument(
    "--measure-gen", action="store_true", help="Measure Generalization properties"
)

parser.add_argument(
    "--queries", type=np.array, default=np.array([]), help="Input noisy queries"
)

args = parser.parse_args()
if args.silent and args.verbose:
    raise ValueError(
        "You cannot choose both --silent and --verbose. These are conflicting options. Choose at most one"
    )

if args.verbose:
    logging.basicConfig(level=logging.DEBUG)
    logging.debug(parser.format_values())
elif not args.silent:
    logging.basicConfig(level=logging.INFO)

if args.all_queries:
    logging.warning(
        "--all-queries option has been chosen. No selection of top queries will occur"
    )

key = random.PRNGKey(args.seed)

dataset = data_sources[args.data_source](
    args.read_file, args.filepath, args.use_subset, args.n, args.d
)
D = np.asarray(dataset.get_dataset())


# update dataset shape
args.n, args.d = D.shape
stat_module = __import__(args.statistic_module)

# First select random k-way marginals from the dataset
kway_attrs = dataset.randomKway(num_kways=args.workload, k=args.k)
kway_compact_queries, _ = dataset.get_queries(kway_attrs)
all_statistic_fn = stat_module.preserve_statistic(kway_compact_queries)
true_statistics = all_statistic_fn(D)

args.epochs = (
    min(args.epochs, np.ceil(len(true_statistics) / args.top_q).astype(np.int32))
    if not args.all_queries
    else 1
)

if args.all_queries:
    # ensure consistency w/ top_q for one-shot case (is this correct?)
    args.top_q = len(true_statistics)

# Initial analysis
print("Number of queries: {}".format(len(true_statistics)))
print("Number of epochs: {}".format(args.epochs))

if args.categorical_consistency:
    print("Categorical consistency")
    feats_csum = np.array([0] + list(dataset.domain.values())).cumsum()
    feats_idx = [
        list(range(feats_csum[i], feats_csum[i + 1]))
        for i in range(len(feats_csum) - 1)
    ]
else:
    feats_idx = None

# Set up the algorithm configuration
algorithm_configuration = RAPConfiguration(
    num_points=args.n,
    num_generated_points=args.n_prime,
    num_dimensions=args.d,
    statistic_function=all_statistic_fn,
    preserve_subset_statistic=stat_module.preserve_subset_statistic,
    get_queries=dataset.get_queries,
    verbose=args.verbose,
    silent=args.silent,
    epochs=args.epochs,
    iterations=args.iterations,
    norm=Norm(args.norm),
    # projection_interval=projection_interval,
    optimizer_learning_rate=args.learning_rate,
    lambda_l1=args.lambda_l1,
    k=args.k,
    top_q=args.top_q,
    use_all_queries=args.all_queries,
    rap_stopping_condition=args.stopping_condition,
    initialize_binomial=args.initialize_binomial,
    feats_idx=feats_idx,
)

# From RAP, main steps!


key, subkey = random.split(key)
rap = RAP(algorithm_configuration, key=key)
# growing number of sanitized statistics to preserve
key, subkey = random.split(subkey)
rap.train(kway_attrs)



all_synth_statistics = all_statistic_fn(rap.D_prime)

print("True statistics:")
print(true_statistics)

print("Synthetic statistics:")
print(all_synth_statistics)

print("Total number of queries:", len(true_statistics))

# Making back into original schema
if args.categorical_consistency:
    Dprime_ohe = rap.generate_rounded_dataset(key)
    all_synth_statistics_ohe = all_statistic_fn(Dprime_ohe)
    max_final_ohe = np.max(np.absolute(true_statistics - all_synth_statistics_ohe))
    l1_final_ohe = np.linalg.norm(
        true_statistics - all_synth_statistics_ohe, ord=1
    ) / float(args.workload)

    l2_final_ohe = np.linalg.norm(true_statistics - all_synth_statistics_ohe, ord=2)
    print("\tFinal rounded max abs error", max_final_ohe)
    print("\tFinal rounded L1 error", l1_final_ohe)
    print("\tFinal rounded L2 error", l2_final_ohe)


# Saving our reconstruction as a set of rows in the original schema
if args.csv_path:
    os.makedirs(args.csv_path, exist_ok=True)

    if args.categorical_consistency:
        Dprime_cat = ohe_to_categorical(Dprime_ohe, feats_idx)
        pd.DataFrame(data=Dprime_cat, columns=list(dataset.domain.keys())).to_csv(
            os.path.join(
                args.csv_path,
                "reconstruction_{}_{}_{}.csv".format(
                    args.data_source,
                    args.workload,
                    args.k,
                ),
            ),
            index=False,
        )
