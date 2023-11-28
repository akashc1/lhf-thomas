#!/usr/bin/env python3
# Copyright (c) Stanford University and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""Initialize the agents.

The following agents are initialized: embedder, generator, oracle,
reward model, generator optimizer, reward optimizer.
"""

import torch
from oracle import OracleNLP, OracleSYN
from my_utils.tfm import TFM, TFM_Pretrained
from my_utils.rembedder import RSentenceTransformer, IdentityEmbbeder
from my_utils.dense import MCD, DeepEnsemble
from my_utils.scaler import MinMaxScalerLayer
from torch.cuda.amp import GradScaler
from my_utils.rational import Rational

from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.nn.parallel import DistributedDataParallel as DDP

def init_agent(rank, config):
    r"""Initialize the agents.

    The oracle can understand the generator vocabulary because it is
    equipped with the generator tokenizer.

    The embedder can understand the generator vocabulary because it is
    equipped with a generator2embedder adaptor, which is a part of actor
    trainable parameters.

    Each agent performs the following mapping:
        Oracle:
            None --> Prompt (SL1 x VSG).
            Generated sequence (q x SL2 x VSG) --> Preference (1).
        Generator: Prompt (SL1 x VSG) --> Generated sequence (SL2 x VSG).
        Embedder: Generator output (SL x VS_G) --> Embedding vector (ES).
        Reward model: Embedding vector (ES) --> Reward (1).

    where:
        SL: sequence_length
        VS_G: vocab_size_generator
        ES: embedding_size
    """
    if config.exp_name == "NLP":
        embedder = RSentenceTransformer(embedder_name=config.embedder_name).to(
            config.device, config.dtype
        )
        config.embedding_size = embedder.embedding_size
        config.vocab_size_embedder = embedder.vocab_size_embedder
        generator = TFM_Pretrained(
            use_SAA=config.use_SAA,
            max_batch_size=config.max_batch_size,
            max_length=config.max_length,
            max_n_restart=config.max_n_restart,
            max_q=config.max_q,
            vocab_size_embedder=config.vocab_size_embedder,
            model_name_or_path=config.generator_name,
        ).to(device=config.device, dtype=config.dtype)
        oracle = OracleNLP(
            scorer_name=config.scorer_name,
            data_name=config.data_name,
            g2i_tok=generator.tokenizer,
        ).to(device=config.device, dtype=config.dtype)
    elif config.exp_name == "SYN":
        embedder = IdentityEmbbeder().to(config.device, config.dtype)
        config.embedding_size = config.max_length
        generator = TFM(
            use_SAA=config.use_SAA,
            max_batch_size=config.max_batch_size,
            max_length=config.max_length,
            max_n_restart=config.max_n_restart,
            max_q=config.max_q,
            bounds_embedder=config.bounds_embedder,
            vocab_size_generator=config.vocab_size_generator,
            d_model=config.d_model,
            nhead=config.nhead,
            num_layers=config.num_layers,
            dim_feedforward=config.dim_feedforward,
            use_dynamic_gradient=config.use_dynamic_gradient,
        ).to(device=config.device, dtype=config.dtype).eval()
        generator.init()
        oracle = OracleSYN(
            g2i_transl=generator.g2e_transl,
            dim_scorer=config.dim_scorer,
            bounds_scorer=config.bounds_scorer,
            vocab_size_generator=config.vocab_size_generator,
            dim_generator=config.dim_generator,
            use_dynamic_gradient=config.use_dynamic_gradient,
        ).to(device=config.device, dtype=config.dtype)
    else:
        raise NotImplementedError

    reward_network_info = dict(
        layers=config.n_layers,
        node_size=config.node_size,
        activation_function=Rational(),
        # activation_function=torch.nn.ELU(),
        last_activation=None,
        dropout_rate=config.dropout_rate,
        n_samples=config.n_samples,
    )
    if config.ABC_method == "MCD":
        Reward_Class = MCD
    elif config.ABC_method == "Ensemble":
        Reward_Class = DeepEnsemble
    else:
        raise NotImplementedError
    
    reward_model = Reward_Class(
        input_size=config.embedding_size,
        output_shape=1,
        info=reward_network_info,
    ).to(config.device, config.dtype)


    generator_optim = torch.optim.Adam(
        generator.parameters(),
        lr=config.generator_lr,
    )
    reward_optim = torch.optim.Adam(
        reward_model.parameters(),
        lr=config.reward_lr,
    )
    generator_scaler = GradScaler()
    reward_scaler = GradScaler()
    reward_output_scaler = MinMaxScalerLayer()
    
    if False:        
        generator = DDP(generator.to(rank), device_ids=[rank])
        reward_model = DDP(reward_model.to(rank), device_ids=[rank])
        
        generator_optim = ZeroRedundancyOptimizer(
            generator.parameters(),
            optimizer_class=torch.optim.Adam,
            lr=config.generator_lr
        )
        
        reward_optim = ZeroRedundancyOptimizer(
            reward_model.parameters(),
            optimizer_class=torch.optim.Adam,
            lr=config.reward_lr
        )
        

    return (
        embedder,
        generator,
        oracle,
        reward_model,
        generator_optim,
        reward_optim,
        generator_scaler,
        reward_scaler,
        reward_output_scaler
    )

class MinMaxScalerLayer(torch.nn.Module):
    def __init__(self, min_val=0.0, max_val=1.0, eps=1e-5):
        super(MinMaxScalerLayer, self).__init__()
        self.min_val = min_val
        self.max_val = max_val
        self.eps = eps

    def forward(self, x):
        x_min = x.min()
        x_max = x.max()
        x_norm = (x - x_min) / (x_max - x_min + self.eps)
        return x_norm * (self.max_val - self.min_val) + self.min_val