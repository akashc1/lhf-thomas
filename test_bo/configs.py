#!/usr/bin/env python3
# Copyright (c) Stanford University and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""Active learning configuration file."""
import os
import torch
import json
import numpy as np
from my_utils.utils import set_seed
from my_utils.reinforce import reinforce
from my_utils.best_of_n import best_of_n
from my_utils.feedme import feedme
from my_utils.naive_bo import naive_bo
from optimize_acqf import optimize_acqf

class Config:
    r"""Configuration class for the active learning experiment."""

    def __init__(self, args, seed) -> None:
        r"""Initialize configuration class."""
        self.seed = seed
        set_seed(seed)
        self.exp_name = args.exp_name
        self.max_length = args.max_length
        self.exp_id = f"exp{args.exp_id}"
        self.algo = args.algo
        if self.algo == "bon":
            self.acqf = best_of_n
        elif self.algo == "fm":
            self.acqf = feedme
        elif self.algo == "bo":
            self.acqf = optimize_acqf
        elif self.algo == "nbo":
            self.acqf = naive_bo
        elif self.algo == "rl":
            self.acqf = reinforce
        else:
            raise NotImplementedError
        
        self.use_SAA = False
        self.generator_lr = 1e-4
        self.reward_lr = 1e-4
        self.entropy_weight = 0
        self.discount = 0.99
        self.use_dynamic_gradient = False
        if args.dynamic:
            self.use_dynamic_gradient = True
        self.bounds_scorer = [-1, 1]
        self.vocab_size_generator = args.vocab_size # 180 # 60000

        self.dtype = torch.float32
        self.device = f"cuda:{args.gpu_id}"
        self.nprocs = 2
        self.use_TS = True
        self.reset_reward = args.reset_reward

        if self.exp_name == "SYN":
            # Embedder params (about its domain and architecture)
            self.bounds_embedder = [-1, 1]
            self.embedder_name = "identity"

            # Scorer params (about its domain and architecture)
            self.bounds_scorer = [-1, 1]
            self.dim_scorer = self.max_length
            self.scorer_name = "SynGP"

            # Reward params (about its architecture)
            self.ABC_method = args.ABC_method
            self.n_samples = 4

            if self.ABC_method == "MCD":
                self.dropout_rate = 0.2
                self.reward_iter = 100
                self.n_layers = 10
                self.use_TS = False

            elif self.ABC_method == "Ensemble":
                self.dropout_rate = 0.0
                self.reward_iter = args.reward_iter
                self.n_layers = 10
                if self.reset_reward:
                    self.n_samples = 1
            
            else:
                raise NotImplementedError

            # Experiment params
            self.q = 2
            self.bo_iter = 100
            self.generator_iter = 10000

            if self.max_length == 1024:
                self.prompt_length = 128
                self.n_init_data = 30000
                self.batch_size = 1
                self.n_restart = 4 # 64
                self.node_size = 2048

                # Generator params (about its architecture)
                self.d_model = 512
                self.nhead = 1
                self.num_layers = 4
                self.dim_feedforward = 256
                self.dim_generator = self.max_length

            elif self.max_length in [2, 10, 100, 1000]:
                # tuned hyperparameters: DO NOT CHANGE
                self.prompt_length = 1
                self.n_init_data = 5
                self.batch_size = 640
                self.n_restart = 1
                self.node_size = 2048
                self.generator_iter = 500 # max(200, self.vocab_size_generator)

                # Generator params (about its architecture)
                if self.vocab_size_generator <= 200:
                    self.d_model = 512
                    self.nhead = 4
                    self.num_layers = 4
                    self.dim_feedforward = 1024
                else:
                    self.batch_size = 1
                    self.n_restart = 1
                    self.d_model = 1024
                    self.nhead = 16
                    self.num_layers = 24
                    self.dim_feedforward = 4096
                
                self.dim_generator = self.max_length

            else:
                raise NotImplementedError

        elif self.exp_name == "NLP":
            # Generator params (about its architecture)
            self.generator_name = "gpt2"

            # Embedder params (about its domain and architecture)
            self.bounds_embedder = [-1, 1]
            self.embedder_name = "all-mpnet-base-v1"

            # Scorer params (about its domain and architecture)
            self.data_name = "imdb"
            self.scorer_name = "lvwerra/distilbert-imdb"

            # Reward params (about its architecture)
            self.n_layers = 3
            self.node_size = 1024
            self.dropout_rate = 0.1

            # Experiment params
            self.q = 2
            self.bo_iter = 1000
            self.reward_iter = 500
            self.rl_iter = 100

            self.n_init_data = 2
            self.batch_size = 2
            self.n_restart = 64
            
            self.eval_batch_size = 16
            
        else:
            raise NotImplementedError

        self.max_q = self.q * 2
        self.max_batch_size = self.batch_size * self.n_init_data
        self.max_n_restart = self.n_restart * 2
        
        if not os.path.exists("results/" + self.exp_id):
            os.makedirs("results/" + self.exp_id)

        # Save config to JSON file
        class dtypeEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, torch.dtype):
                # might wanna add np, jnp types too?
                    return str(obj)
                return json.JSONEncoder.default(self, obj)
            
        with open(f"results/{self.exp_id}/config.json", "w") as f:
            dict = self.__dict__.copy()
            dict.pop("acqf", None)
            json.dump(dict, f, indent=4, cls=dtypeEncoder)
