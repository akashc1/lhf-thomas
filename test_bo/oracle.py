#!/usr/bin/env python3
# Copyright (c) Stanford University and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""Implement an oracle to evaluate active learning on sequence generation.

This file contains the oracle that is used to evaluate the performance of the
active learning algorithm.
"""

from abc import ABC
import torch
from torch import Tensor
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification as AM4SC
from my_utils.synthfunc import SynGP
from botorch.test_functions import Ackley, Branin

class Oracle(ABC):
    r"""Abstract class for an oracle that allows query preference.

    1.The input to the oracle is sequence of $P$ token $a_{1:P}$,
        each of which comes from the *generator domain* $\mathbb{C}_g$.
    3.Each oracle is equipped with a score function. The input to the
        score function is a sequence of $Q$ token $b_{1:Q}, each of
        which comes from the *score function domain* $\mathbb{C}_s$.
    2.To allow the score function to score $a_{1:P}$, $a_{1:P}$ is
        first mapped to a sequence of $T$ token $i_{1:T}$, each of which
        comes from the *intermediate domain* $\mathbb{C}_i$. The
        intermediate will be mapped to the score function domain
        after that:

        $$ a_{1:P} \mapsto i_{1:T} \mapsto b_{1:Q} $$.

        The first mapping is done by the *generator-to-intermediate
        translator*. The second mapping is done by the *intermediate-to-
        score translator*.
    """

    def __init__(self) -> None:
        r"""Check for required attributes."""
        required_attributes = [
            "g2i_transl",
            "i2s_transl",
            "scorer",
            "postprocess_score",
            "samples_from_generator_domain",
            "vocab_size_generator",
        ]
        for i in required_attributes:
            if not hasattr(self, i):
                raise ValueError(f"{i} attribute is required for oracle.")

    def __call__(self, query: Tensor):
        r"""Return the preferred option among `q` choices.

        Args:
            query (Tensor): a one-hot tensor with dimention
                `... x q x max_length x vocab_size_generator`

        Returns:
            A tensor with shape `... x 1` of index of option
                with the highest score among `q` options.
        """

        # query_shape = query.shape
        # query = query.reshape(-1, *query_shape[-2:])
        # >>> (... * q) x max_length x vocab_size_generator

        outputs = self.compute_score(query)
        # >>> (... * q)

        # outputs = outputs.reshape(*query_shape[:-2])
        # >>> ... x q

        return outputs.argmax(-1)
        # >>> ...

    def compute_score(self, query: Tensor):
        """Compute the score of each options.

        Args:
            query (Tensor): a one-hot tensor queries generated from
                generator with dimension
                `... * q x max_length x vocab_size_generator`

        Returns:
            A tensor with shape `... * q` of score.
        """
        query = self.i2s_transl(self.g2i_transl(query))
        return self.postprocess_score(self.scorer(query))
        # >>> ... * q

    def get_prompt(self, batch_size: int, prompt_length: int) -> Tensor:
        r"""Return a batch of prompts.

        Args:
            batch_size (int): number of prompts to return.
            prompt_length (int): length of each prompt.

        Returns:
            A one-hot tensor of shape
                `prompt_length x batch_size x vocab_size_generator`.
                We use this shape to be consistent with the input
                shape required for PyTorch's transformer.
        """
        if len(self.samples_from_generator_domain) < batch_size:
            raise ValueError("Not enough data to meet batch size.")

        # permute index of data in generator domain
        ids = torch.randperm(len(self.samples_from_generator_domain))

        prompt = []
        i = 0
        while len(prompt) < batch_size:
            item = self.samples_from_generator_domain[ids[i]]
            if len(item) >= prompt_length:
                prompt.append(item[:prompt_length])
            i = i + 1

        prompt = torch.stack(prompt)
        # >>> batch_size x prompt_length

        prompt = prompt.permute(1, 0)
        # >>> prompt_length x batch_size

        if self.use_dynamic_gradient:
            return torch.nn.functional.one_hot(
                prompt, num_classes=self.vocab_size_generator
            ).to(device=self.device, dtype=self.dtype)
            # >>> prompt_len x batch_size x vocab_size_generator

        else:
            return prompt.to(device=self.device, dtype=torch.int32)
            # >>> prompt_len x batch_size

    def to(self, device, dtype):
        r"""Move the scorer to a device and change the data type."""
        self.dtype = dtype
        self.device = device
        self.scorer = self.scorer.to(device=device, dtype=dtype)
        return self


class OracleNLP(Oracle):
    r"""NLP oracle with an underlying semantic classifier as scorer."""

    def __init__(self, **kwargs: dict) -> None:
        r"""Initialize required attributes."""
        for i in ["g2i_tok", "scorer_name", "data_name", "use_dynamic_gradient"]:
            if i not in kwargs:
                raise ValueError(f"{i} is required for NLP oracle.")

        self.use_dynamic_gradient = kwargs["use_dynamic_gradient"]

        # 1. Generator to intermediate translator
        g2i_tok = kwargs["g2i_tok"]
        self.g2i_transl = lambda x: g2i_tok.batch_decode(x.argmax(-1))

        # 2. Intermediate to score translator
        i2s_tok = AutoTokenizer.from_pretrained(
            kwargs["scorer_name"],
            padding_side="right",
        )
        self.i2s_transl = lambda x: i2s_tok(
            x, padding=True, return_tensors="pt"
        )["input_ids"].to(self.device)

        # 3. Score function
        self.scorer = AM4SC.from_pretrained(kwargs["scorer_name"])
        self.scorer.eval()
        for param in self.scorer.parameters():
            param.requires_grad = False

        # 4. Postprocess score
        self.postprocess_score = lambda x: x.logits.argmax(-1)

        # 5. Samples from generator domain
        self.samples_from_generator_domain = g2i_tok(
            load_dataset(kwargs["data_name"])["train"]["text"],
            truncation=True,
            padding=True,
            return_tensors="pt",
        )["input_ids"]
        self.vocab_size_generator = g2i_tok.vocab_size

        super().__init__()


class OracleSYN(Oracle):
    r"""Synthetic oracle with underlying GP as scorer."""

    def __init__(self, **kwargs) -> None:
        r"""Initialize required attributes."""
        required_keys = [
            "g2i_transl",
            "dim_scorer",
            "bounds_scorer",
            "vocab_size_generator",
            "dim_generator",
            "use_dynamic_gradient",
        ]
        for i in required_keys:
            if i not in kwargs:
                raise ValueError(f"{i} is required for SYN oracle.")

        self.use_dynamic_gradient = kwargs["use_dynamic_gradient"]

        # 1. Generator to intermediate translator
        self.g2i_transl = kwargs["g2i_transl"]

        # 2. Intermediate to score translator
        self.i2s_transl = lambda x: x

        # 3. Score function
        self.scorer = SynGP(dim=kwargs["dim_scorer"])
        # self.scorer = Ackley(dim=kwargs["dim_scorer"])
        # self.scorer = Branin(dim=kwargs["dim_scorer"])
        for i in [0, 1]:
            self.scorer.bounds[i, :].fill_(kwargs["bounds_scorer"][i])

        # 4. Postprocess score
        self.postprocess_score = lambda x: x.squeeze(-1)

        # 5. Samples from generator domain
        class DiscreteDomainSampler:
            def __init__(self, vocab_size, dim) -> None:
                self.vocab_size = vocab_size
                self.dim = dim

            def __getitem__(self, _):
                return torch.randint(high=self.vocab_size, size=(self.dim,))

            def __len__(self):
                return 100000

        self.samples_from_generator_domain = DiscreteDomainSampler(
            vocab_size=kwargs["vocab_size_generator"],
            dim=kwargs["dim_generator"],
        )
        self.vocab_size_generator = kwargs["vocab_size_generator"]

        super().__init__()


if __name__ == "__main__":
    oracle = OracleSYN()
    a = oracle.model(torch.tensor([[2, 5], [3, 4]]))
    print(a)
    b = oracle.get_prompt(10)
    print(b)
