#!/usr/bin/env python3
# Copyright (c) Stanford University and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""Implement the Transformer-based generator.

This module implement Transformer generator that allows 
generating samples that are differentiable with respecto to
the input token. We also implement Sample Average Approximation
(SAA) to reduce the variance to facilitate training.
"""

from abc import ABC, abstractmethod
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from my_utils.cat_with_base import OneHotCategoricalSTVarianceReduction
from my_utils.utils import load_actor_with_logit
from my_utils.pos_enc import PositionalEncoding
# from loralib import Linear, Embedding
from torch.nn import Linear, Embedding

# r = 1024 # r for quantization

class TFM_Base(ABC, nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @abstractmethod
    def forward(self, tgt):
        raise NotImplementedError

    @abstractmethod
    def g2e_transl(self, tgt):
        raise NotImplementedError

    def compute_next_token_dist(self, input_tensors, cache=None):
        r"""
        Predict the next token given the input tokens and the model.

        Args:
            input_tensors: A float one-hot tensor with shape 
                (sequence_length, batch_size, num_categories).

        Returns:
            int: The ID of the predicted next token.
        """
        # Pass the input tensor through the model to get the logits
        logits, cache = self.forward(input_tensors, cache)
        # >>> sequence_length x batch x num_categories

        # Get the logits for the last token in the sequence
        last_token_logits = logits[-1, :, :]

        # Apply the softmax function to get the probabilities of the next token
        probs = F.softmax(last_token_logits, dim=-1)

        if self.use_dynamic_gradient:
            dist_class = OneHotCategoricalSTVarianceReduction
        else:
            dist_class = torch.distributions.Categorical

        return dist_class(probs=probs), cache

    def generate(
        self,
        input_tensors: Tensor,
        n_restart: int,
        q: int,
        max_length: int,
        log_probs_tokens: Tensor = None
    ):
        r"""
        Generate a sequence of tokens given an input sequence.
        
        Args:
            input_tensors (torch.Tensor): A long tensor with shape
                (sequence_length, batch_size, num_categories).
            n_restart (int): Number of restarts.
            q (int): Number of samples per restart.
            max_length (int): Maximum length of the generated sequence.
         
        """
        prompt_length, batch_size = input_tensors.shape[:2]
        if self.use_SAA:
            assert n_restart <= self.max_n_restart
            assert q <= self.max_q
            assert max_length <= self.max_length
            assert batch_size <= self.max_batch_size
        
        entropies = []
        log_probs = []

        # Initialize the generated_tokens tensor
        repeat_shape = [1]*len(input_tensors.shape)
        repeat_shape[1] = n_restart*q
        generated_tokens = input_tensors.repeat(*repeat_shape)
        generated_probs = input_tensors.repeat(*repeat_shape).long()
        num_categories = self.vocab_size_generator
        if not self.use_dynamic_gradient:
            generated_probs = torch.nn.functional.one_hot(
                                    generated_probs, 
                                    num_classes=self.vocab_size_generator
                                    ).to(torch.float32)
        else:
            generated_probs = generated_probs.to(torch.float32)
        # >>> sequence_length x (n_restart * q * batch_size) (x num_categories)
        cache = None

        for i in range(prompt_length, max_length):
            dists, cache = self.compute_next_token_dist(generated_tokens, cache)
            # >>> n_restart * q * batch_size (x num_categories)

            if self.use_SAA:
                base_samples = self.base_samples[:max_length, :n_restart*q*batch_size]
                # >>> max_length x n_restart * q * batch_size

                base_samples_i = base_samples[i]
                # >>> n_restart * q * batch_size

                next_tokens = dists.rsample_with_base_sample(base_samples_i)
                # >>> n_restart * q * batch_size x num_categories
            else:
                if self.use_dynamic_gradient:
                    next_tokens = dists.rsample()
                    # >>> n_restart * q * batch_size x num_categories
                else:
                    next_tokens = dists.sample()
                    # >>> n_restart * q * batch_size

            generated_tokens = torch.cat([generated_tokens, next_tokens[None]], dim=0)
            generated_probs = torch.cat([generated_probs, dists.probs[None]], dim=0)
            # >>> (sequence_length + 1) 
            # ... x n_restart * q * batch_size 
            # ... x num_categories
            
            entropies.append(dists.entropy())
            if log_probs_tokens is not None:
                log_probs.append(dists.log_prob(log_probs_tokens[i]))
            else:
                log_probs.append(dists.log_prob(next_tokens))

        if self.use_dynamic_gradient:
            outputs = generated_tokens.permute(1, 0, 2)
            # >>> (n_restart * q * batch_size) x max_length x num_categories
            outputs = outputs.reshape(
                n_restart, q, batch_size, max_length, num_categories
            )
            outputs = outputs.permute(2, 0, 1, 3, 4)
            # >>> batch_size x n_restart x q x max_length x num_categories
        else:
            outputs = generated_tokens.transpose(1, 0)
            # >>> (n_restart * q * batch_size) x max_length 
            outputs = outputs.reshape(
                n_restart, q, batch_size, max_length
            )
            outputs = outputs.permute(2, 0, 1, 3)
            # >>> batch_size x n_restart x q x max_length
            
            outputs = outputs.to(torch.int32)
            
        outputs_probs = generated_probs.permute(1, 0, 2)
        # >>> (n_restart * q * batch_size) x max_length x num_categories
        outputs_probs = outputs_probs.reshape(
            n_restart, q, batch_size, max_length, num_categories
        )
        outputs_probs = outputs_probs.permute(2, 0, 1, 3, 4)
        # >>> batch_size x n_restart x q x max_length x num_categories

        comp_len = max_length - prompt_length

        quantities = []
        for quant in [entropies, log_probs]:
            quant = torch.stack(quant, dim=0)
            # >>> comp_len x (n_restart * q * batch_size)

            quant = quant.reshape(comp_len, n_restart, q, batch_size)
            # >>> comp_len x n_restart x q x batch_size

            quant = quant.permute(3, 1, 2, 0)
            # >>> batch_size x n_restart x q x comp_len

            quantities.append(quant)
        
        return outputs, outputs_probs, *quantities

    def reset_base_samples(self):
        r"""Reset base samples for SAA."""
        self.base_samples = torch.rand(
            self.max_length, 
            self.max_batch_size*self.max_n_restart*self.max_q
        )


class TFM(TFM_Base):
    def __init__(
            self,
            use_SAA: bool,
            max_batch_size: int,
            max_length: int,
            max_n_restart: int,
            max_q: int,
            bounds_embedder: tuple,
            vocab_size_generator: int,
            d_model: int,
            nhead: int,
            num_layers: int,
            dim_feedforward: int,
            use_dynamic_gradient: bool,
        ) -> None:
        r"""
        Transformer-based generator.
        
        Args:
            use_SAA (bool): Whether to use Sample Average Approximation.
            max_batch_size (int): Maximum batch size.
            max_length (int): Maximum length of the generated sequence.
            max_n_restart (int): Maximum number of restarts.
            max_q (int): Maximum number of samples per restart.
            vocab_size_generator (int): Size of the vocabulary.
            d_model (int): Dimensionality of the embedding.
            nhead (int): Number of attention heads.
            num_layers (int): Number of layers.
            dim_feedforward (int): Dimensionality of the feedforward layer.
        """
        super(TFM, self).__init__()

        self.max_length = max_length
        self.max_n_restart = max_n_restart
        self.max_q = max_q
        self.max_batch_size = max_batch_size
        self.vocab_size_generator = vocab_size_generator
        self.use_SAA = use_SAA
        if use_SAA:
            self.reset_base_samples()

        # transformer
        self.d_model = d_model
        self.use_dynamic_gradient = use_dynamic_gradient
        if use_dynamic_gradient:
            self.embedding = Linear(
                vocab_size_generator, d_model, bias=False, # r=r
            )
            
        else:
            self.embedding = Embedding(
                vocab_size_generator, d_model, # r=r
            )
            
        self.positional_encoding = PositionalEncoding(d_model)
        tfm_layer = CausalTransformerDecoderLayer(d_model, nhead, dim_feedforward)
        self.tfm = CausalTransformerDecoder(tfm_layer, num_layers)
        self.fc = Linear(
            d_model, vocab_size_generator, #r=r
        )

        # generator2embedder adapter
        self.bounds_embedder = bounds_embedder
        self.vocab_size_generator = vocab_size_generator
        self.range_size = (self.bounds_embedder[1] - self.bounds_embedder[0])
        self.range_size = self.range_size / self.vocab_size_generator
        midpoints = []
        for i in range(self.vocab_size_generator):
            midpoint = self.bounds_embedder[0] + self.range_size * (i + 0.5)
            rand = torch.rand(1).item() * self.range_size - self.range_size / 2
            midpoint = midpoint + rand

            # clip to bounds
            midpoint = max(midpoint, self.bounds_embedder[0])
            midpoint = min(midpoint, self.bounds_embedder[1])

            midpoints.append(midpoint)

        midpoints = torch.tensor(
            midpoints
        ).reshape(1, -1)
            
        if use_dynamic_gradient:
            self.g2e_transl_layer = nn.Linear(
                vocab_size_generator, 1, bias=False, #r=r
            )
            self.g2e_transl_layer.weight.data = midpoints
        else:
            self.g2e_transl_layer = nn.Embedding(
                vocab_size_generator, 1, _weight=midpoints.T, #r=r
            )

        for param in self.g2e_transl_layer.parameters():
            param.requires_grad = False

    def forward(self, tgt, cache):
        r"""
        Args:
            tgt (torch.Tensor): A one-hot tensor with shape
                (sequence_length, batch_size, num_categories).
        """
        tgt = self.embedding(tgt) * (tgt.size(1) ** 0.5)
        tgt = self.positional_encoding(tgt)
        memory = None
        
        output, cache = self.tfm(tgt, memory, cache)
        return self.fc(output), cache
        # >>> sequence_length x batch_size x vocab_size_embedder

    def g2e_transl(self, tgt):
        return self.g2e_transl_layer(tgt).squeeze(dim=-1)

    def init(self):
        r"""Initialize the neural network so that the output are uniform."""

        print("Initializing the model...")
        batch_size = 100
        epoch = 1000
        lr = 1e-3
        optim = torch.optim.Adam(self.parameters(), lr=lr)
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        for i in range(epoch):
            sequence_length = torch.randint(
                low=1, high=self.max_length, size=(1,)
            ).item()
            tgt = torch.randint(
                low=0, high=self.vocab_size_generator,
                size=(sequence_length, batch_size),
                device=device,
            )
            if self.use_dynamic_gradient:
                tgt = F.one_hot(tgt, num_classes=self.vocab_size_generator).to(dtype)
    
            dist, _ = self.compute_next_token_dist(tgt)
            loss = abs(dist.probs - 1.0 / self.vocab_size_generator).mean()
            loss.backward()
            optim.step()
            optim.zero_grad()



class TFM_Pretrained(TFM_Base):
    def __init__(
        self,
        use_SAA: bool,
        max_batch_size: int,
        max_length: int,
        max_n_restart: int,
        max_q : int,
        vocab_size_embedder: int,
        model_name_or_path: str,
    ) -> None:
        
        super(TFM_Pretrained, self).__init__()
        
        self.max_length = max_length
        self.max_n_restart = max_n_restart
        self.max_q = max_q
        self.max_batch_size = max_batch_size
        self.use_SAA = use_SAA
        if use_SAA:
            self.reset_base_samples()

        self.model, self.tokenizer = load_actor_with_logit(
            model_name_or_path
        )
        self.g2e_transl_layer = Linear(
            self.tokenizer.vocab_size, 
            vocab_size_embedder, 
            bias=False, 
            # r=r
        )

        new_pes = []
        for pe in [
            self.model.transformer.wte
        ]:
            vocab_size_generator, hidden_dim = pe.weight.shape
            new_pe = Linear(
                vocab_size_generator,
                hidden_dim, 
                bias=False, 
                device=self.model.device, 
                # r=r
            )
            new_pe.weight = nn.parameter.Parameter(pe.weight.T)
            new_pes.append(new_pe)

        self.model.transformer.wte = new_pes[0]

    def forward(self, tgt):
        r"""
        Compute logit of the next token given previous tokens.
        
        tgt: (sequence_length, batch_size, num_categories)
        
        """
        
        input_tgt = tgt.permute(1, 0, 2)
        # >>> batch_size x sequence_length x num_categories
        
        inputs_embeds = self.model._modules['transformer'].wte(input_tgt)
        # >>> batch_size x sequence_length x hidden_dim
        
        output = self.model(inputs_embeds=inputs_embeds)
        # >>> batch_size x sequence_length x num_categories
        
        output = output.logits.permute(1, 0, 2)
        # >>> sequence_length x batch_size x num_categories
    
        return output
        # >>> sequence_length x batch_size x num_categories

    def g2e_transl(self, tgt):
        return self.g2e_transl_layer(tgt)
        # >>> sequence_length x batch_size x vocab_size_embedder


import torch
import torch.nn as nn
from typing import Optional
from torch import Tensor


class CausalTransformerDecoder(nn.TransformerDecoder):
    """Implementation of a transformer decoder based on torch implementation but
    more efficient. The difference is that it doesn't need to recompute the
    embeddings of all the past decoded tokens but instead uses a cache to
    store them. This makes use of the fact that the attention of a decoder is
    causal, so new predicted tokens don't affect the old tokens' embedding bc
    the corresponding attention cells are masked.
    The complexity goes from seq_len^3 to seq_len^2.

    This only happens in eval mode.
    In training mode, teacher forcing makes these optimizations unnecessary. Hence the
    Decoder acts like a regular nn.TransformerDecoder (except that the attention tgt
    masks are handled for you).
    """

    def forward(
        self,
        tgt: Tensor,
        memory: Optional[Tensor] = None,
        cache: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            tgt (Tensor): current_len_output x bsz x hidden_dim
            memory (Tensor): len_encoded_seq x bsz x hidden_dim
            cache (Optional[Tensor]):
                n_layers x (current_len_output - 1) x bsz x hidden_dim
                If current_len_output == 1, nothing is cached yet, so cache
                should be None. Same if the module is in training mode.
            others (Optional[Tensor]): see official documentations
        Returns:
            output (Tensor): current_len_output x bsz x hidden_dim
            cache (Optional[Tensor]): n_layers x current_len_output x bsz x hidden_dim
                Only returns it when module is in eval mode (no caching in training)
        """

        output = tgt

        new_token_cache = []
        for i, mod in enumerate(self.layers):
            output = mod(output, memory)
            new_token_cache.append(output)
            if cache is not None:
                output = torch.cat([cache[i], output], dim=0)

        if cache is not None:
            new_cache = torch.cat([cache, torch.stack(new_token_cache, dim=0)], dim=1)
        else:
            new_cache = torch.stack(new_token_cache, dim=0)

        return output, new_cache


class CausalTransformerDecoderLayer(nn.TransformerDecoderLayer):
    def forward(
        self,
        tgt: Tensor,
        memory: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            see CausalTransformerDecoder
        Returns:
            Tensor:
                If training: embedding of the whole layer: seq_len x bsz x hidden_dim
                If eval mode: embedding of last token: 1 x bsz x hidden_dim
        """
        # This part is adapted from the official Pytorch implementation
        # So that only the last token gets modified and returned.

        tgt_last_tok = tgt[-1:, :, :]

        # self attention part
        tmp_tgt = self.self_attn(
            tgt_last_tok,
            tgt,
            tgt,
            attn_mask=None,  # not needed because we only care about the last token
            key_padding_mask=tgt_key_padding_mask,
        )[0]
        tgt_last_tok = tgt_last_tok + self.dropout1(tmp_tgt)
        tgt_last_tok = self.norm1(tgt_last_tok)

        # encoder-decoder attention
        if memory is not None:
            tmp_tgt = self.multihead_attn(
                tgt_last_tok,
                memory,
                memory,
                attn_mask=memory_mask,
                key_padding_mask=memory_key_padding_mask,
            )[0]
            tgt_last_tok = tgt_last_tok + self.dropout2(tmp_tgt)
            
        tgt_last_tok = self.norm2(tgt_last_tok)

        # final feed-forward network
        tmp_tgt = self.linear2(
            self.dropout(self.activation(self.linear1(tgt_last_tok)))
        )
        tgt_last_tok = tgt_last_tok + self.dropout3(tmp_tgt)
        tgt_last_tok = self.norm3(tgt_last_tok)
        return tgt_last_tok


def generate_square_subsequent_mask(sz: int, device: str = "cpu") -> torch.Tensor:
    """ Generate the attention mask for causal decoding """
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    ).to(device=device)
    return mask

    
if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    use_SAA = False
    model_name_or_path = "gpt2"
    vocab_size_generator = 50257
    vocab_size_embedder = 10000
    d_model = 512
    nhead = 8
    num_layers = 6
    dim_feedforward = 2048
    max_length = 10
    n_restart = 5
    q = 3
    max_batch_size = 10
    max_n_restart = 10
    max_q = 10
    dtype = torch.float32

    exps = dict(
        exp1=dict(use_SAA=True, G=TFM),
        exp2=dict(use_SAA=False, G=TFM),
        exp3=dict(use_SAA=True, G=TFM_Pretrained),
        exp4=dict(use_SAA=False, G=TFM_Pretrained),
    )
    for exp in exps.values():
        use_SAA = exp["use_SAA"]
        G = exp["G"]

        if G == TFM:
            print("Testing TFM")
            generator = G(
                use_SAA=use_SAA,
                max_batch_size=max_batch_size,
                max_length=max_length,
                max_n_restart=max_n_restart,
                max_q=max_q,
                vocab_size_generator=vocab_size_generator,
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers,
                dim_feedforward=dim_feedforward,
            ).to(device)

        elif G == TFM_Pretrained:
            print("Testing TFM_Pretrained")
            generator = G(
                use_SAA=use_SAA,
                max_batch_size=max_batch_size,
                max_length=max_length,
                max_n_restart=max_n_restart,
                max_q=max_q,
                vocab_size_embedder=vocab_size_embedder,
                model_name_or_path=model_name_or_path,
            ).to(device)

        # Set the model to evaluation mode
        generator.eval()

        # Example input sequence of token IDs
        input_tensors = torch.tensor(
            [[10, 15, 25], [1,2,3]],
            dtype=torch.long, device=device,
        ).T
        # >>> sequence_length x batch_size

        # convert to one-hot
        input_tensors = F.one_hot(
            input_tensors, num_classes=vocab_size_generator
        ).to(dtype=dtype)
        # >>> sequence_length x batch_size x num_categories

        generated_tokens, _ = generator.generate(
            input_tensors=input_tensors, 
            n_restart=n_restart, 
            q=q,
            max_length=max_length
        )
        generated_tokens = generated_tokens.argmax(dim=-1)

        print("Generated sequence:", generated_tokens)
