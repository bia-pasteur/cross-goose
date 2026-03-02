from abc import ABC, abstractmethod
from typing import List

import torch
from torch import nn

from crossgoose.model.modules import linblock


class FlowFunction(ABC, nn.Module):

    @abstractmethod
    def forward(self, e0, et):
        raise NotImplementedError


class FlowAttention(FlowFunction):
    def __init__(
        self,
        embedding_dim: int,
        key_dim: int,
        nb_key: int,
        value_dim: int = 2
    ):
        super().__init__()
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.nb_key = nb_key
        self.embedding_dim = embedding_dim

        self.query_module = linblock(embedding_dim, key_dim)
        self.key_module = linblock(embedding_dim, nb_key*key_dim)
        self.value_module = linblock(embedding_dim, nb_key*value_dim)

    def forward(self, e0, et):

        n, f = e0.shape
        assert tuple(et.shape) == (n, f), (et.shape, (n, f))

        query = self.query_module(e0).view(n, 1, self.key_dim)
        key = self.key_module(et).view(n, self.nb_key, self.key_dim)
        value = self.value_module(et).view(n, self.nb_key, self.value_dim)

        # https://docs.pytorch.org/docs/2.4/generated/torch.nn.functional.scaled_dot_product_attention.html
        return nn.functional.scaled_dot_product_attention(  # pylint: disable=E1102
            query=query,
            key=key,
            value=value
        ).squeeze(dim=-2)


class FlowLinear(FlowFunction):
    def __init__(
        self,
        embedding_dim: int,
        hidden_dims: List[int],
        value_dim: int = 2,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim

        all_dims = hidden_dims + [value_dim]
        modules = []
        last_dim = embedding_dim*2
        for dim in all_dims:
            modules.append(
                linblock(
                    in_channels=last_dim,
                    out_channels=dim
                ))
            last_dim = dim

        self.transform = nn.Sequential(
            *modules
        )

    def forward(self, e0, et):
        # n, f = e0.shape
        # assert tuple(et.shape) == (n, f)
        e0et = torch.concat([e0, et], dim=-1)

        return self.transform(e0et)
