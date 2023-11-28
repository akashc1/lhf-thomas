import numpy as np
import torch
import torch.nn as nn
from torch.distributions import OneHotCategoricalStraightThrough
# from loralib import Linear
from torch.nn import Linear
r = 16

class MultiOutputDropout(nn.Module):
    def __init__(self, p=0.5):
        super(MultiOutputDropout, self).__init__()
        self.p = p
        if self.p < 1:
            self.multiplier_ = 1.0 / (1.0 - self.p)
        else:
            self.multiplier_ = 0.0
        self.selected = None

    def forward(self, input):
        if not self.training:
            return input

        selected = torch.empty_like(input, device=input.device)
        self.selected = (selected.uniform_(0, 1) > self.p).float()
        self.selected[..., 1:, :] = self.selected[..., 0:1, :]
        
        return self.selected * input * self.multiplier_


def make_dense(
    input_size,
    output_shape,
    info,
):
    activation_function = info['activation_function']
    last_activation = info['last_activation']
    # self.dropout = MultiOutputDropout(p=info['dropout_rate'])
    dropout = nn.Dropout(p=info['dropout_rate'])
    node_size = info['node_size']
    layers = info['layers']
    model = [Linear(
        input_size, node_size, #r=r
    )]
    model += [activation_function, dropout]
    for i in range(layers-1):
        model += [Linear(
            node_size, node_size, #r=r
        )]
        model += [activation_function, dropout]

    model += [Linear(
        node_size, int(np.prod(output_shape)), #r=r
    )]
    if last_activation is not None:
        model += [last_activation]

    return nn.Sequential(*model)


class MCD(nn.Module):
    def __init__(self, input_size, output_shape, info):
        """
        :param output_shape: tuple containing shape of expected output
        :param input_size: size of input features
        :param info: dict containing num of hidden layers, size of hidden layers,
         activation function, output distribution etc.
        """
        super().__init__()
        self.n_samples = info['n_samples']
        self.model = make_dense(input_size, output_shape, info)
        self.sm = torch.nn.Softmax(dim=-2)

    def forward(self, input, *args, **kwargs):
        return self.model(input).squeeze(-1)

    def posterior_function(self, input, *args, **kwargs):
        input = input[None].expand(self.n_samples, *input.shape)
        return self.model(input)


class DeepEnsemble(nn.Module):
    def __init__(self, input_size, output_shape, info):
        super().__init__()
        self.n_samples = info['n_samples']
        models = []
        for _ in range(self.n_samples):
            models.append(make_dense(input_size, output_shape, info))
        self.models = nn.ModuleList(models)

    def forward(self, input, TS_idx=None):
        if TS_idx is not None:
            return self.models[TS_idx](input).squeeze(-1)
        else:
            return torch.stack([m(input) for m in self.models]).squeeze(-1)

    def posterior_function(self, input, TS_idx=None):
        if TS_idx is not None:
            return self.models[TS_idx](input)
        else:
            return torch.stack([m(input) for m in self.models])

# class NeuralActor(DenseModel, nn.Module):
#     def forward(self, states):
#         action_probs =  self.model(states)
#         return OneHotCategoricalStraightThrough(probs=action_probs)
    
#     def generate(self, prompt):
#         output = self.forward(prompt).sample().long()
#         output = torch.cat([prompt, output], dim=-2)
#         return output
