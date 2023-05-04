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
        # Initialize the synthetic dataset
        self.D_prime = self.__initialize_synthetic_dataset(key)

        self.statistics_l1 = []
        self.statistics_max = []
        self.means_l1 = []
        self.means_max = []
        self.max_errors = []
        self.l2_errors = []
        self.losses = []

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

        if self.args.projection_interval:
            # If we are projecting into [a,b], start with a dataset in range.
            interval = self.args.projection_interval
            if len(interval) != 2 or interval.projection_max <= interval.projection_min:
                raise ValueError(
                    "Must input interval in the form '--project a b' to project into [a,b], b>a"
                )
            return self.__compute_initial_dataset(
                SyntheticInitializationOptions.RANDOM_INTERVAL, key
            )
        else:
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

        print("jit_loss_fn")

        ord_norm = norm_mapping[self.args.norm]

        # I THINK WE'D WANT TO EDIT THE BELOW VALUES
        @jit
        def compute_loss_fn(
            synthetic_dataset: np.DeviceArray, target_statistics: np.DeviceArray
        ) -> np.DeviceArray:
            extraLoss = 0
            print(len(target_statistics))            
            print(target_statistics[0])
        
            # if target_statistics[0] == 0:
            #     extraLoss = 1000
            if self.args.norm is Norm.LOG_EXP:
                return np.log(
                    np.exp(statistic_fn(synthetic_dataset) - target_statistics).sum()
                ) + self.args.lambda_l1 * np.linalg.norm(synthetic_dataset, 1)
            else:
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
        # print("get_update_fn")

        opt_init, opt_update, get_params = optimizer(learning_rate)
        opt_state = opt_init(self.D_prime)

        @jit
        def update(synthetic_dataset, target_statistics, state):
            """Compute the gradient and update the parameters"""
            value, grads = value_and_grad(loss_fn)(synthetic_dataset, target_statistics)
            state = opt_update(0, grads, state)
            return get_params(state), state, value

        return update, opt_state


    def __clip_array(self, array: np.DeviceArray) -> np.DeviceArray:
        # print("clip_array")

        if self.args.projection_interval:
            projection_min, projection_max = self.args.projection_interval
            return np.clip(array, projection_min, projection_max)
        else:
            return array


    def train(
        self, dataset: np.DeviceArray, k_way_attributes: Any, key: np.DeviceArray
    ) -> None:
        print("train")

        # THESE TRUE STATISTICS WILL TECHNICALLY BE OUR NOISY ONES / INPUT
        true_statistics = self.args.statistic_function(dataset)
        # for i in range(len(true_statistics)):
        #     true_statistics = true_statistics.at[i].set(1.0)
        print("true statistics")
        print(true_statistics)

        sanitized_queries = np.array([])
        target_statistics = np.array([])
        k_way_queries, total_queries = self.args.get_queries(k_way_attributes, N=-1)
        k_way_queries = np.asarray(k_way_queries)

        for epoch in range(self.args.epochs):
            query_errs = np.abs(
                self.args.statistic_function(self.D_prime) - true_statistics
            )
           
            if self.args.use_all_queries:
                selected_indices = np.arange(len(k_way_queries))
            else:
                selected_indices = select_noisy_q(query_errs, sanitized_queries, 
                                                  self.args.top_q)

            selected_queries = k_way_queries.take(selected_indices, axis=0)
            current_statistic_fn = self.args.preserve_subset_statistic(selected_queries)


            target_statistics = np.concatenate(
                [
                    target_statistics,
                    current_statistic_fn(dataset)
                ]
            )

            sanitized_queries = np.asarray(
                np.append(sanitized_queries, selected_indices), dtype=np.int32
            )

            curr_queries = k_way_queries[sanitized_queries]
            curr_statistic_fn = self.args.preserve_subset_statistic(
                np.asarray(curr_queries)
            )

            target_statistics = self.__clip_array(target_statistics)
            # print("hereee")
            # print(target_statistics)
            # target_statistics = target_statistics.at[2].set(1)

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
                self.statistics_l1.append(
                    np.mean(np.absolute(target_statistics - synthetic_statistics))
                )
                self.statistics_max.append(
                    np.amax(np.absolute(target_statistics - synthetic_statistics))
                )
                self.means_l1.append(
                    np.mean(np.absolute(np.mean(dataset, 0) - np.mean(self.D_prime, 0)))
                )
                self.means_max.append(
                    np.amax(np.absolute(np.mean(dataset, 0) - np.mean(self.D_prime, 0)))
                )
                all_synth_statistics = self.args.statistic_function(self.D_prime)
                self.max_errors.append(
                    np.max(np.absolute(true_statistics - all_synth_statistics))
                )
                self.l2_errors.append(
                    np.linalg.norm(true_statistics - all_synth_statistics, ord=2)
                )
                self.losses.append(loss)

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
