# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
import time
import logging
from typing import Tuple, Any, Callable

import numpy as np_orig
from jax import numpy as np, random, jit, value_and_grad
from jax.example_libraries import optimizers

from utils_data import sparsemax_project, randomized_rounding, select_noisy_q
from .constants import SyntheticInitializationOptions, norm_mapping, Norm
from .rap_configuration import RAPConfiguration


class RAP:
    def __init__(self, args: RAPConfiguration, key: np.DeviceArray):
        self.args = args
        self.start_time = time.time()
        self.D_prime = self.__initialize_synthetic_dataset(key)
        self.feats_idx = args.feats_idx
        if self.args.verbose:
            logging.basicConfig(level=logging.DEBUG)
        elif not self.args.silent:
            logging.basicConfig(level=logging.INFO)


    def __compute_initial_dataset(
        self, selection: SyntheticInitializationOptions, key: np.DeviceArray
    ) -> np.DeviceArray:
        """
        Function that computes D_prime based on input
        :param selection: the type of synthetic data initialization
        :param: key: key to generate random numbers with
        :return: initial hypothesis of synthetic data
        """

        shape = (self.args.num_generated_points, self.args.num_dimensions)
        random_initial = random.uniform(key=key, shape=shape)

        if selection is SyntheticInitializationOptions.RANDOM:
            return 2 * (random_initial - 0.5)
        elif selection is SyntheticInitializationOptions.RANDOM_INTERVAL:
            interval = self.args.projection_interval
            return (
                (interval.projection_max - interval.projection_min) * random_initial
            ) + interval.projection_min
        elif selection is SyntheticInitializationOptions.RANDOM_BINOMIAL:
            return np_orig.array(
                [
                    np_orig.random.binomial(1, p=self.args.probs)
                    for _ in range(self.args.num_generated_points)
                ],
                dtype=float,
            )
        else:
            raise ValueError(
                "Supported selections are ",
                [
                    member.value
                    for _, member in SyntheticInitializationOptions.__members__.items()
                ],
            )


    def __initialize_synthetic_dataset(self, key: np.DeviceArray):
        """
        Function that
        :param key: key to generate random numbers with
        :return:
        """
        if self.args.initialize_binomial:
            return self.__compute_initial_dataset(
                SyntheticInitializationOptions.RANDOM_BINOMIAL, key
            )
        else:
            return self.__compute_initial_dataset(
                SyntheticInitializationOptions.RANDOM, key
            )
            

    def __jit_loss_fn(
        self, statistic_fn: Callable[[np.DeviceArray], np.DeviceArray]
    ) -> Callable[[np.DeviceArray, np.DeviceArray], np.DeviceArray]:

        ord_norm = norm_mapping[self.args.norm]

        @jit
        def compute_loss_fn(
            synthetic_dataset: np.DeviceArray, target_statistics: np.DeviceArray
        ) -> np.DeviceArray:
            return np.linalg.norm(
                    statistic_fn(synthetic_dataset) - target_statistics, ord=ord_norm
                ) + self.args.lambda_l1 * np.linalg.norm(synthetic_dataset, 1)

        return compute_loss_fn


    def __get_update_function(
        self,
        learning_rate: float,
        optimizer: Callable[..., optimizers.Optimizer],
        loss_fn: Callable[[np.DeviceArray, np.DeviceArray], np.DeviceArray],
    ) -> Tuple[
        Callable[[np.DeviceArray, np.DeviceArray], np.DeviceArray], np.DeviceArray
    ]:

        opt_init, opt_update, get_params = optimizer(learning_rate)
        opt_state = opt_init(self.D_prime)

        @jit
        def update(synthetic_dataset, target_statistics, state):
            """Compute the gradient and update the parameters"""
            value, grads = value_and_grad(loss_fn)(synthetic_dataset, target_statistics)
            state = opt_update(0, grads, state)
            return get_params(state), state, value

        return update, opt_state


    # MOST IMPORTANT FOR RECONSTRUCTION
    def train(
        self, k_way_attributes: Any
    ) -> None:

        # NOISY STATS GO HERE:
        # actually correct = [0.,0.2, 0.1, 0.,  0.7, 0.,  0.,  0. ]
        true_statistics = np.array([0.,0.25, 0, 0., 0.75, 0.,  0.,  0. ])


        sanitized_queries = np.array([])
        target_statistics = np.array([])
        k_way_queries, total_queries = self.args.get_queries(k_way_attributes, N=-1)
        k_way_queries = np.asarray(k_way_queries)

        for epoch in range(self.args.epochs):

            # USE OUR HINTS HERE
            hints_dict = {2: 0.1} # saying the value at index 2 should be 0.1
            for key, val in hints_dict.items():
                true_statistics = true_statistics.at[key].set(val)

            # L2 Error
            query_errs = np.abs(
                (self.args.statistic_function(self.D_prime) - true_statistics) ** 2
            )

            # Apply our loss here:
            for key, val in hints_dict.items():
                hint_q_error = query_errs[key]
                if hint_q_error != 0.0:
                    query_errs = query_errs.at[key].set(1000 ** hint_q_error)

            if self.args.use_all_queries:
                selected_indices = np.arange(len(k_way_queries))
            else:
                selected_indices = select_noisy_q(query_errs, sanitized_queries, 
                                                  self.args.top_q)


            # selects the worst current query to update on next round
            valuesOfIndices = []
            for i in selected_indices:
                valuesOfIndices.append(true_statistics[i])
                break

            target_statistics = np.concatenate(
                [
                    target_statistics,
                    np.asarray(valuesOfIndices)
                ]
            )

    
            sanitized_queries = np.asarray(
                np.append(sanitized_queries, selected_indices), dtype=np.int32
            )
            curr_queries = k_way_queries[sanitized_queries]
            curr_statistic_fn = self.args.preserve_subset_statistic(
                np.asarray(curr_queries)
            )


            loss_fn = self.__jit_loss_fn(curr_statistic_fn)
            previous_loss = np.inf

            optimizer_learning_rate = (
                self.args.optimizer_learning_rate
            )  
            update, opt_state = self.__get_update_function(
                optimizer_learning_rate, optimizers.adam, loss_fn
            )

            for iteration in range(self.args.iterations):
                self.D_prime, opt_state, loss = update(
                    self.D_prime, target_statistics, opt_state
                )
                if self.feats_idx:
                    self.D_prime = sparsemax_project(self.D_prime, self.feats_idx)
                else:
                    self.D_prime = self.__clip_array(self.D_prime)

                synthetic_statistics = curr_statistic_fn(self.D_prime)

                all_synth_statistics = self.args.statistic_function(self.D_prime)

                # Stop early if we made no progress this round.
                if loss >= previous_loss - self.args.rap_stopping_condition:
                    logging.info("Stopping early at iteration {}".format(iteration))
                    break

                previous_loss = loss



    def generate_rounded_dataset(self, key, oversample=None):
        if not oversample:
            oversample = self.args.num_points // self.args.num_generated_points

        return randomized_rounding(
            D=self.D_prime, feats_idx=self.feats_idx, key=key, oversample=oversample
        )
