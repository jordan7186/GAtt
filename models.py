"""
This file contains the GAT models used in the experiments.
"""

# Import the required libraries
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_add_pool, global_max_pool

# From the source code from GATConv
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import NoneType  # noqa
from torch_geometric.typing import (
    Adj,
    OptPairTensor,
    OptTensor,
    Size,
    SparseTensor,
    torch_sparse,
    PairTensor,
)
from torch_geometric.utils import (
    add_self_loops,
    is_torch_sparse_tensor,
    remove_self_loops,
    softmax,
)
from torch_geometric.utils.sparse import set_sparse_value

"""
For GATv2
"""
import typing

if typing.TYPE_CHECKING:
    from typing import overload
else:
    from torch.jit import _overload_method as overload

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import Adj, OptTensor, SparseTensor, torch_sparse
from torch_geometric.utils import (
    add_self_loops,
    batched_negative_sampling,
    dropout_edge,
    is_undirected,
    negative_sampling,
    remove_self_loops,
    softmax,
    to_undirected,
)


class SuperGATConv_mask(MessagePassing):
    att_x: OptTensor
    att_y: OptTensor

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        add_self_loops: bool = True,
        bias: bool = True,
        attention_type: str = "MX",
        neg_sample_ratio: float = 0.5,
        edge_sample_ratio: float = 1.0,
        is_undirected: bool = False,
        **kwargs,
    ):
        kwargs.setdefault("aggr", "add")
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.attention_type = attention_type
        self.neg_sample_ratio = neg_sample_ratio
        self.edge_sample_ratio = edge_sample_ratio
        self.is_undirected = is_undirected

        assert attention_type in ["MX", "SD"]
        assert 0.0 < neg_sample_ratio and 0.0 < edge_sample_ratio <= 1.0

        self.lin = Linear(
            in_channels, heads * out_channels, bias=False, weight_initializer="glorot"
        )

        if self.attention_type == "MX":
            self.att_l = Parameter(torch.Tensor(1, heads, out_channels))
            self.att_r = Parameter(torch.Tensor(1, heads, out_channels))
        else:  # self.attention_type == 'SD'
            self.register_parameter("att_l", None)
            self.register_parameter("att_r", None)

        self.att_x = self.att_y = None  # x/y for self-supervision

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin.reset_parameters()
        glorot(self.att_l)
        glorot(self.att_r)
        zeros(self.bias)

    def forward(
        self,
        x: Tensor,
        edge_index: Adj,
        neg_edge_index: OptTensor = None,
        batch: OptTensor = None,
        return_attention_weights: bool = False,
        mask_edge: Tuple = None,
        post=False,
    ) -> Tensor:
        N, H, C = x.size(0), self.heads, self.out_channels

        if self.add_self_loops:
            if isinstance(edge_index, SparseTensor):
                edge_index = torch_sparse.fill_diag(edge_index, 1.0)
            else:
                edge_index, _ = remove_self_loops(edge_index)
                edge_index, _ = add_self_loops(edge_index, num_nodes=N)

        x = self.lin(x).view(-1, H, C)

        # propagate_type: (x: Tensor)
        if mask_edge is not None:
            mask_edge_index = (edge_index[0] == mask_edge[0]) & (
                edge_index[1] == mask_edge[1]
            )
        else:
            mask_edge_index = None
        out = self.propagate(edge_index, x=x, size=None, mask=mask_edge_index)

        # Always return "unmasked" attention coefficients
        if return_attention_weights:
            att = self.get_attention(
                edge_index_i=edge_index[1],
                x_i=x[edge_index[1]],
                x_j=x[edge_index[0]],
                num_nodes=x.size(0),
                return_logits=False,
            )
            att_package = (edge_index, att)

        if self.training:
            if isinstance(edge_index, SparseTensor):
                col, row, _ = edge_index.coo()
                edge_index = torch.stack([row, col], dim=0)
            pos_edge_index = self.positive_sampling(edge_index)

            pos_att = self.get_attention(
                edge_index_i=pos_edge_index[1],
                x_i=x[pos_edge_index[1]],
                x_j=x[pos_edge_index[0]],
                num_nodes=x.size(0),
                return_logits=True,
            )

            if neg_edge_index is None:
                neg_edge_index = self.negative_sampling(edge_index, N, batch)

            neg_att = self.get_attention(
                edge_index_i=neg_edge_index[1],
                x_i=x[neg_edge_index[1]],
                x_j=x[neg_edge_index[0]],
                num_nodes=x.size(0),
                return_logits=True,
            )

            self.att_x = torch.cat([pos_att, neg_att], dim=0)
            self.att_y = self.att_x.new_zeros(self.att_x.size(0))
            self.att_y[: pos_edge_index.size(1)] = 1.0

        if self.concat is True:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out = out + self.bias

        if return_attention_weights:
            return out, att_package
        return out

    def message(
        self,
        edge_index_i: Tensor,
        x_i: Tensor,
        x_j: Tensor,
        size_i: Optional[int],
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        alpha = self.get_attention(
            edge_index_i, x_i, x_j, num_nodes=size_i, return_logits=True
        )
        alpha = F.leaky_relu(alpha, self.negative_slope)
        if mask is not None:
            alpha[mask] = 0
        alpha = softmax(alpha, edge_index_i, num_nodes=size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.view(-1, self.heads, 1)

    def negative_sampling(
        self, edge_index: Tensor, num_nodes: int, batch: OptTensor = None
    ) -> Tensor:

        num_neg_samples = int(
            self.neg_sample_ratio * self.edge_sample_ratio * edge_index.size(1)
        )

        if not self.is_undirected and not is_undirected(
            edge_index, num_nodes=num_nodes
        ):
            edge_index = to_undirected(edge_index, num_nodes=num_nodes)

        if batch is None:
            neg_edge_index = negative_sampling(
                edge_index, num_nodes, num_neg_samples=num_neg_samples
            )
        else:
            neg_edge_index = batched_negative_sampling(
                edge_index, batch, num_neg_samples=num_neg_samples
            )

        return neg_edge_index

    def positive_sampling(self, edge_index: Tensor) -> Tensor:
        pos_edge_index, _ = dropout_edge(
            edge_index, p=1.0 - self.edge_sample_ratio, training=self.training
        )
        return pos_edge_index

    def get_attention(
        self,
        edge_index_i: Tensor,
        x_i: Tensor,
        x_j: Tensor,
        num_nodes: Optional[int],
        return_logits: bool = False,
    ) -> Tensor:

        if self.attention_type == "MX":
            logits = (x_i * x_j).sum(dim=-1)
            if return_logits:
                return logits

            alpha = (x_j * self.att_l).sum(-1) + (x_i * self.att_r).sum(-1)
            alpha = alpha * logits.sigmoid()

        else:  # self.attention_type == 'SD'
            alpha = (x_i * x_j).sum(dim=-1) / math.sqrt(self.out_channels)
            if return_logits:
                return alpha

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index_i, num_nodes=num_nodes)
        return alpha

    def get_attention_loss(self) -> Tensor:
        r"""Computes the self-supervised graph attention loss."""
        if not self.training:
            return torch.tensor([0], device=self.lin.weight.device)

        return F.binary_cross_entropy_with_logits(
            self.att_x.mean(dim=-1),
            self.att_y,
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.in_channels}, "
            f"{self.out_channels}, heads={self.heads}, "
            f"type={self.attention_type})"
        )


class SuperGAT_L2_intervention(torch.nn.Module):
    def __init__(
        self, in_channels, hidden_channels, out_channels, heads=1, att_type="MX"
    ):
        super().__init__()

        self.conv1 = SuperGATConv_mask(
            in_channels=in_channels,
            out_channels=hidden_channels,
            heads=heads,
            dropout=0.6,
            attention_type=att_type,
            edge_sample_ratio=0.8,
            is_undirected=True,
        )
        self.conv2 = SuperGATConv_mask(
            in_channels=hidden_channels * heads,
            out_channels=out_channels,
            heads=heads,
            concat=False,
            dropout=0.6,
            attention_type=att_type,
            edge_sample_ratio=0.8,
            is_undirected=True,
        )

    #  post is a deprecated parameter, not used
    def forward(
        self,
        x,
        edge_index,
        return_att=False,
        mask_edge=None,
        post=False,
    ):
        if return_att:
            x = F.dropout(x, p=0.6, training=self.training)
            x, att1 = self.conv1(
                x, edge_index, return_attention_weights=True, mask_edge=mask_edge
            )
            x = F.elu(x)
            att_loss = self.conv1.get_attention_loss()
            x = F.dropout(x, p=0.6, training=self.training)
            x, att2 = self.conv2(
                x, edge_index, return_attention_weights=True, mask_edge=mask_edge
            )
            att_loss += self.conv2.get_attention_loss()
            self.att = [att1, att2]
            return F.log_softmax(x, dim=-1), att_loss
        else:
            x = F.dropout(x, p=0.6, training=self.training)
            x = self.conv1(x, edge_index, mask_edge=mask_edge)
            x = F.elu(x)
            att_loss = self.conv1.get_attention_loss()
            x = F.dropout(x, p=0.6, training=self.training)
            x = self.conv2(x, edge_index, mask_edge=mask_edge)
            att_loss += self.conv2.get_attention_loss()
            return F.log_softmax(x, dim=-1), att_loss


class SuperGAT_L3_intervention(torch.nn.Module):
    def __init__(
        self, in_channels, hidden_channels, out_channels, heads=1, att_type="MX"
    ):
        super().__init__()

        self.conv1 = SuperGATConv_mask(
            in_channels=in_channels,
            out_channels=hidden_channels,
            heads=heads,
            dropout=0.6,
            attention_type=att_type,
            edge_sample_ratio=0.8,
            is_undirected=True,
        )
        self.conv2 = SuperGATConv_mask(
            in_channels=hidden_channels * heads,
            out_channels=hidden_channels,
            heads=heads,
            dropout=0.6,
            attention_type=att_type,
            edge_sample_ratio=0.8,
            is_undirected=True,
        )
        self.conv3 = SuperGATConv_mask(
            in_channels=hidden_channels * heads,
            out_channels=out_channels,
            heads=heads,
            concat=False,
            dropout=0.6,
            attention_type=att_type,
            edge_sample_ratio=0.8,
            is_undirected=True,
        )

    #  post is a deprecated parameter, not used
    def forward(
        self,
        x,
        edge_index,
        return_att=False,
        mask_edge=None,
        post=False,
    ):
        if return_att:
            x = F.dropout(x, p=0.6, training=self.training)
            x, att1 = self.conv1(
                x, edge_index, return_attention_weights=True, mask_edge=mask_edge
            )
            x = F.elu(x)
            att_loss = self.conv1.get_attention_loss()
            x = F.dropout(x, p=0.6, training=self.training)
            x, att2 = self.conv2(
                x, edge_index, return_attention_weights=True, mask_edge=mask_edge
            )
            att_loss += self.conv2.get_attention_loss()
            x = F.dropout(x, p=0.6, training=self.training)
            x, att3 = self.conv3(
                x, edge_index, return_attention_weights=True, mask_edge=mask_edge
            )
            att_loss += self.conv3.get_attention_loss()
            self.att = [att1, att2, att3]
            return F.log_softmax(x, dim=-1), att_loss
        else:
            x = F.dropout(x, p=0.6, training=self.training)
            x = self.conv1(x, edge_index, mask_edge=mask_edge)
            x = F.elu(x)
            att_loss = self.conv1.get_attention_loss()
            x = F.dropout(x, p=0.6, training=self.training)
            x = self.conv2(x, edge_index, mask_edge=mask_edge)
            att_loss += self.conv2.get_attention_loss()
            x = F.dropout(x, p=0.6, training=self.training)
            x = self.conv3(x, edge_index, mask_edge=mask_edge)
            att_loss += self.conv3.get_attention_loss()
            return F.log_softmax(x, dim=-1), att_loss


# Define the GAT model, with 2 hidden layers
class GAT_L2(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads, dropout=0):
        super().__init__()
        self.conv1 = GATConv(
            in_channels=in_channels,
            out_channels=hidden_channels,
            heads=heads,
            concat=True,
            dropout=dropout,
            add_self_loops=True,
        )
        self.conv2 = GATConv(
            in_channels=hidden_channels * heads,
            out_channels=out_channels,
            heads=1,
            concat=False,
            add_self_loops=True,
        )
        self.att = []

    def forward(self, x, edge_index, return_att: bool = False):
        if return_att:
            x, att1 = self.conv1(x, edge_index, return_attention_weights=True)
            x = F.elu(x)
            x, att2 = self.conv2(x, edge_index, return_attention_weights=True)
            self.att = [att1, att2]
        else:
            x = self.conv1(x, edge_index)
            x = F.elu(x)
            x = self.conv2(x, edge_index)
        return x


# Define the GAT model, with 3 hidden layers
class GAT_L3(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads, dropout=0):
        super().__init__()
        self.conv1 = GATConv(
            in_channels=in_channels,
            out_channels=hidden_channels,
            heads=heads,
            concat=True,
            dropout=dropout,
            add_self_loops=True,
        )
        self.conv2 = GATConv(
            in_channels=hidden_channels * heads,
            out_channels=hidden_channels,
            heads=heads,
            concat=True,
            dropout=dropout,
            add_self_loops=True,
        )
        self.conv3 = GATConv(
            in_channels=hidden_channels * heads,
            out_channels=out_channels,
            heads=1,
            concat=False,
            dropout=dropout,
            add_self_loops=True,
        )
        self.att = []

    def forward(self, x, edge_index, return_att: bool = False):
        if return_att:
            x, att1 = self.conv1(x, edge_index, return_attention_weights=True)
            x = F.elu(x)
            x, att2 = self.conv2(x, edge_index, return_attention_weights=True)
            x = F.elu(x)
            x, att3 = self.conv3(x, edge_index, return_attention_weights=True)
            self.att = [att1, att2, att3]
        else:
            x = self.conv1(x, edge_index)
            x = F.elu(x)
            x = self.conv2(x, edge_index)
            x = F.elu(x)
            x = self.conv3(x, edge_index)
        return x


class GAT_karate(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GAT_karate, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=1)
        self.conv2 = GATConv(hidden_channels, out_channels, heads=1)

    def forward(self, x, edge_index):
        x, att1 = self.conv1(x, edge_index, return_attention_weights=True)
        x = F.elu(x)
        x, att2 = self.conv2(x, edge_index, return_attention_weights=True)
        return x, (att1, att2)


class GATConv_mask(MessagePassing):
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        add_self_loops: bool = True,
        edge_dim: Optional[int] = None,
        fill_value: Union[float, Tensor, str] = "mean",
        bias: bool = True,
        **kwargs,
    ):
        kwargs.setdefault("aggr", "add")
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.edge_dim = edge_dim
        self.fill_value = fill_value

        # In case we are operating in bipartite graphs, we apply separate
        # transformations 'lin_src' and 'lin_dst' to source and target nodes:
        if isinstance(in_channels, int):
            self.lin_src = Linear(
                in_channels,
                heads * out_channels,
                bias=False,
                weight_initializer="glorot",
            )
            self.lin_dst = self.lin_src
        else:
            self.lin_src = Linear(
                in_channels[0], heads * out_channels, False, weight_initializer="glorot"
            )
            self.lin_dst = Linear(
                in_channels[1], heads * out_channels, False, weight_initializer="glorot"
            )

        # The learnable parameters to compute attention coefficients:
        self.att_src = Parameter(torch.empty(1, heads, out_channels))
        self.att_dst = Parameter(torch.empty(1, heads, out_channels))

        if edge_dim is not None:
            self.lin_edge = Linear(
                edge_dim, heads * out_channels, bias=False, weight_initializer="glorot"
            )
            self.att_edge = Parameter(torch.empty(1, heads, out_channels))
        else:
            self.lin_edge = None
            self.register_parameter("att_edge", None)

        if bias and concat:
            self.bias = Parameter(torch.empty(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin_src.reset_parameters()
        self.lin_dst.reset_parameters()
        if self.lin_edge is not None:
            self.lin_edge.reset_parameters()
        glorot(self.att_src)
        glorot(self.att_dst)
        glorot(self.att_edge)
        zeros(self.bias)

    def forward(
        self,
        x: Union[Tensor, OptPairTensor],
        edge_index: Adj,
        edge_attr: OptTensor = None,
        size: Size = None,
        return_attention_weights=None,
        mask_edge=None,
        post: bool = False,
    ):
        H, C = self.heads, self.out_channels

        # We first transform the input node features. If a tuple is passed, we
        # transform source and target node features via separate weights:
        if isinstance(x, Tensor):
            assert x.dim() == 2, "Static graphs not supported in 'GATConv'"
            x_src = x_dst = self.lin_src(x).view(-1, H, C)
        else:  # Tuple of source and target node features:
            x_src, x_dst = x
            assert x_src.dim() == 2, "Static graphs not supported in 'GATConv'"
            x_src = self.lin_src(x_src).view(-1, H, C)
            if x_dst is not None:
                x_dst = self.lin_dst(x_dst).view(-1, H, C)

        x = (x_src, x_dst)

        # Next, we compute node-level attention coefficients, both for source
        # and target nodes (if present):
        alpha_src = (x_src * self.att_src).sum(dim=-1)
        alpha_dst = None if x_dst is None else (x_dst * self.att_dst).sum(-1)
        alpha = (alpha_src, alpha_dst)

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                # We only want to add self-loops for nodes that appear both as
                # source and target nodes:
                num_nodes = x_src.size(0)
                if x_dst is not None:
                    num_nodes = min(num_nodes, x_dst.size(0))
                num_nodes = min(size) if size is not None else num_nodes
                edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
                edge_index, edge_attr = add_self_loops(
                    edge_index,
                    edge_attr,
                    fill_value=self.fill_value,
                    num_nodes=num_nodes,
                )
            elif isinstance(edge_index, SparseTensor):
                if self.edge_dim is None:
                    edge_index = torch_sparse.set_diag(edge_index)
                else:
                    raise NotImplementedError(
                        "The usage of 'edge_attr' and 'add_self_loops' "
                        "simultaneously is currently not yet supported for "
                        "'edge_index' in a 'SparseTensor' form"
                    )

        # edge_updater_type: (alpha: OptPairTensor, edge_attr: OptTensor)
        alpha = self.edge_updater(edge_index, alpha=alpha, edge_attr=edge_attr)

        # Identify indices of edges that need to be masked
        if mask_edge is not None:
            mask_edge_index = (edge_index[0] == mask_edge[0]) & (
                edge_index[1] == mask_edge[1]
            )
            if post is False:
                alpha[mask_edge_index] = 0

        # propagate_type: (x: OptPairTensor, alpha: Tensor)
        if post is False:
            out = self.propagate(edge_index, x=x, alpha=alpha, size=size, mask=None)
        else:
            out = self.propagate(
                edge_index, x=x, alpha=alpha, size=size, mask=mask_edge_index
            )

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out = out + self.bias

        if isinstance(return_attention_weights, bool):
            if isinstance(edge_index, Tensor):
                if is_torch_sparse_tensor(edge_index):
                    # TODO TorchScript requires to return a tuple
                    adj = set_sparse_value(edge_index, alpha)
                    return out, (adj, alpha)
                else:
                    return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout="coo")
        else:
            return out

    def edge_update(
        self,
        alpha_j: Tensor,
        alpha_i: OptTensor,
        edge_attr: OptTensor,
        index: Tensor,
        ptr: OptTensor,
        size_i: Optional[int],
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        # Given edge-level attention coefficients for source and target nodes,
        # we simply need to sum them up to "emulate" concatenation:
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
        if index.numel() == 0:
            return alpha
        if edge_attr is not None and self.lin_edge is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            edge_attr = self.lin_edge(edge_attr)
            edge_attr = edge_attr.view(-1, self.heads, self.out_channels)
            alpha_edge = (edge_attr * self.att_edge).sum(dim=-1)
            alpha = alpha + alpha_edge

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        if mask is not None:
            alpha[mask] = 0
        return alpha

    def message(self, x_j: Tensor, alpha: Tensor) -> Tensor:
        return alpha.unsqueeze(-1) * x_j

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.in_channels}, "
            f"{self.out_channels}, heads={self.heads})"
        )


class GATv2Conv_mask(MessagePassing):
    _alpha: OptTensor

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        add_self_loops: bool = True,
        edge_dim: Optional[int] = None,
        fill_value: Union[float, Tensor, str] = "mean",
        bias: bool = True,
        share_weights: bool = False,
        **kwargs,
    ):
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.edge_dim = edge_dim
        self.fill_value = fill_value
        self.share_weights = share_weights

        if isinstance(in_channels, int):
            self.lin_l = Linear(
                in_channels,
                heads * out_channels,
                bias=bias,
                weight_initializer="glorot",
            )
            if share_weights:
                self.lin_r = self.lin_l
            else:
                self.lin_r = Linear(
                    in_channels,
                    heads * out_channels,
                    bias=bias,
                    weight_initializer="glorot",
                )
        else:
            self.lin_l = Linear(
                in_channels[0],
                heads * out_channels,
                bias=bias,
                weight_initializer="glorot",
            )
            if share_weights:
                self.lin_r = self.lin_l
            else:
                self.lin_r = Linear(
                    in_channels[1],
                    heads * out_channels,
                    bias=bias,
                    weight_initializer="glorot",
                )

        self.att = Parameter(torch.empty(1, heads, out_channels))

        if edge_dim is not None:
            self.lin_edge = Linear(
                edge_dim, heads * out_channels, bias=False, weight_initializer="glorot"
            )
        else:
            self.lin_edge = None

        if bias and concat:
            self.bias = Parameter(torch.empty(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)

        self._alpha = None

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()
        if self.lin_edge is not None:
            self.lin_edge.reset_parameters()
        glorot(self.att)
        zeros(self.bias)

    @overload
    def forward(
        self,
        x: Union[Tensor, PairTensor],
        edge_index: Adj,
        edge_attr: OptTensor = None,
        return_attention_weights: NoneType = None,
    ) -> Tensor:
        pass

    @overload
    def forward(  # noqa: F811
        self,
        x: Union[Tensor, PairTensor],
        edge_index: Tensor,
        edge_attr: OptTensor = None,
        return_attention_weights: bool = None,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        pass

    @overload
    def forward(  # noqa: F811
        self,
        x: Union[Tensor, PairTensor],
        edge_index: SparseTensor,
        edge_attr: OptTensor = None,
        return_attention_weights: bool = None,
    ) -> Tuple[Tensor, SparseTensor]:
        pass

    def forward(  # noqa: F811
        self,
        x: Union[Tensor, PairTensor],
        edge_index: Adj,
        edge_attr: OptTensor = None,
        return_attention_weights: Optional[bool] = None,
        mask_edge: Tuple = None,
        post: bool = False,
    ) -> Union[
        Tensor,
        Tuple[Tensor, Tuple[Tensor, Tensor]],
        Tuple[Tensor, SparseTensor],
    ]:
        H, C = self.heads, self.out_channels

        x_l: OptTensor = None
        x_r: OptTensor = None
        if isinstance(x, Tensor):
            assert x.dim() == 2
            x_l = self.lin_l(x).view(-1, H, C)
            if self.share_weights:
                x_r = x_l
            else:
                x_r = self.lin_r(x).view(-1, H, C)
        else:
            x_l, x_r = x[0], x[1]
            assert x[0].dim() == 2
            x_l = self.lin_l(x_l).view(-1, H, C)
            if x_r is not None:
                x_r = self.lin_r(x_r).view(-1, H, C)

        assert x_l is not None
        assert x_r is not None

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                num_nodes = x_l.size(0)
                if x_r is not None:
                    num_nodes = min(num_nodes, x_r.size(0))
                edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
                edge_index, edge_attr = add_self_loops(
                    edge_index,
                    edge_attr,
                    fill_value=self.fill_value,
                    num_nodes=num_nodes,
                )
            elif isinstance(edge_index, SparseTensor):
                if self.edge_dim is None:
                    edge_index = torch_sparse.set_diag(edge_index)
                else:
                    raise NotImplementedError(
                        "The usage of 'edge_attr' and 'add_self_loops' "
                        "simultaneously is currently not yet supported for "
                        "'edge_index' in a 'SparseTensor' form"
                    )

        # Identify indices of edges that need to be masked
        if mask_edge is not None:
            mask_edge_index = (edge_index[0] == mask_edge[0]) & (
                edge_index[1] == mask_edge[1]
            )

        # propagate_type: (x: PairTensor, edge_attr: OptTensor)
        if mask_edge is None:
            out = self.propagate(
                edge_index, x=(x_l, x_r), edge_attr=edge_attr, mask=None
            )
        else:
            out = self.propagate(
                edge_index,
                x=(x_l, x_r),
                edge_attr=edge_attr,
                mask=mask_edge_index,
                post=post,
            )

        alpha = self._alpha
        assert alpha is not None
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out = out + self.bias

        if isinstance(return_attention_weights, bool):
            if isinstance(edge_index, Tensor):
                if is_torch_sparse_tensor(edge_index):
                    # TODO TorchScript requires to return a tuple
                    adj = set_sparse_value(edge_index, alpha)
                    return out, (adj, alpha)
                else:
                    return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout="coo")
        else:
            return out

    def message(
        self,
        x_j: Tensor,
        x_i: Tensor,
        edge_attr: OptTensor,
        index: Tensor,
        ptr: OptTensor,
        size_i: Optional[int],
        mask: Optional[Tensor] = None,
        post: bool = False,
    ) -> Tensor:
        x = x_i + x_j

        if edge_attr is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            assert self.lin_edge is not None
            edge_attr = self.lin_edge(edge_attr)
            edge_attr = edge_attr.view(-1, self.heads, self.out_channels)
            x = x + edge_attr

        x = F.leaky_relu(x, self.negative_slope)
        alpha = (x * self.att).sum(dim=-1)
        if mask is not None and post is False:
            alpha[mask] = 0
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        if mask is not None and post:
            alpha[mask] = 0
        return x_j * alpha.unsqueeze(-1)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.in_channels}, "
            f"{self.out_channels}, heads={self.heads})"
        )


# Since the previous models have defined on GAT_L2 and GAT_L3, we need a way to
# copy the model parameters from those models to the ones defined from
# GATConv_intervention.


class GAT_L2_intervention(torch.nn.Module):
    def __init__(
        self, in_channels, hidden_channels, out_channels, heads, dropout=0, **kwargs
    ):
        super().__init__()
        self.conv1 = GATConv_mask(
            in_channels=in_channels,
            out_channels=hidden_channels,
            heads=heads,
            concat=True,
            dropout=dropout,
            add_self_loops=True,
            **kwargs,
        )
        self.conv2 = GATConv_mask(
            in_channels=hidden_channels * heads,
            out_channels=out_channels,
            heads=1,
            concat=False,
            add_self_loops=True,
            **kwargs,
        )
        self.att = []

    def forward(
        self,
        x,
        edge_index,
        return_att: bool = False,
        mask_edge: Tuple = None,
        post: bool = False,
    ):
        if return_att:
            x, att1 = self.conv1(
                x,
                edge_index,
                return_attention_weights=True,
                mask_edge=mask_edge,
                post=post,
            )
            x = F.elu(x)
            x, att2 = self.conv2(
                x,
                edge_index,
                return_attention_weights=True,
                mask_edge=mask_edge,
                post=post,
            )
            self.att = [att1, att2]
        else:
            x = self.conv1(x, edge_index, mask_edge=mask_edge, post=post)
            x = F.elu(x)
            x = self.conv2(x, edge_index, mask_edge=mask_edge, post=post)
        return x


class GAT_L3_intervention(torch.nn.Module):
    def __init__(
        self, in_channels, hidden_channels, out_channels, heads, dropout=0, **kwargs
    ):
        super().__init__()
        self.conv1 = GATConv_mask(
            in_channels=in_channels,
            out_channels=hidden_channels,
            heads=heads,
            concat=True,
            dropout=dropout,
            add_self_loops=True,
            **kwargs,
        )
        self.conv2 = GATConv_mask(
            in_channels=hidden_channels * heads,
            out_channels=hidden_channels,
            heads=heads,
            concat=True,
            dropout=dropout,
            add_self_loops=True,
            **kwargs,
        )
        self.conv3 = GATConv_mask(
            in_channels=hidden_channels * heads,
            out_channels=out_channels,
            heads=1,
            concat=False,
            dropout=dropout,
            add_self_loops=True,
            **kwargs,
        )
        self.att = []

    def forward(
        self,
        x,
        edge_index,
        return_att: bool = False,
        mask_edge: Tuple = None,
        post: bool = False,
    ):
        if return_att:
            x, att1 = self.conv1(
                x,
                edge_index,
                return_attention_weights=True,
                mask_edge=mask_edge,
                post=post,
            )
            x = F.elu(x)
            x, att2 = self.conv2(
                x,
                edge_index,
                return_attention_weights=True,
                mask_edge=mask_edge,
                post=post,
            )
            x = F.elu(x)
            x, att3 = self.conv3(
                x,
                edge_index,
                return_attention_weights=True,
                mask_edge=mask_edge,
                post=post,
            )
            self.att = [att1, att2, att3]
        else:
            x = self.conv1(x, edge_index, mask_edge=mask_edge, post=post)
            x = F.elu(x)
            x = self.conv2(x, edge_index, mask_edge=mask_edge, post=post)
            x = F.elu(x)
            x = self.conv3(x, edge_index, mask_edge=mask_edge, post=post)
        return x


class GATv2_L2_intervention(torch.nn.Module):
    def __init__(
        self, in_channels, hidden_channels, out_channels, heads, dropout=0, **kwargs
    ):
        super().__init__()
        self.conv1 = GATv2Conv_mask(
            in_channels=in_channels,
            out_channels=hidden_channels,
            heads=heads,
            concat=True,
            dropout=dropout,
            add_self_loops=True,
            **kwargs,
        )
        self.conv2 = GATv2Conv_mask(
            in_channels=hidden_channels * heads,
            out_channels=out_channels,
            heads=1,
            concat=False,
            add_self_loops=True,
            **kwargs,
        )
        self.att = []

    def forward(
        self,
        x,
        edge_index,
        return_att: bool = False,
        mask_edge: Tuple = None,
        post: bool = False,
    ):
        if return_att:
            x, att1 = self.conv1(
                x,
                edge_index,
                return_attention_weights=True,
                mask_edge=mask_edge,
                post=post,
            )
            x = F.elu(x)
            x, att2 = self.conv2(
                x,
                edge_index,
                return_attention_weights=True,
                mask_edge=mask_edge,
                post=post,
            )
            self.att = [att1, att2]
        else:
            x = self.conv1(x, edge_index, mask_edge=mask_edge, post=post)
            x = F.elu(x)
            x = self.conv2(x, edge_index, mask_edge=mask_edge, post=post)
        return x


class GATv2_L3_intervention(torch.nn.Module):
    def __init__(
        self, in_channels, hidden_channels, out_channels, heads, dropout=0, **kwargs
    ):
        super().__init__()
        self.conv1 = GATv2Conv_mask(
            in_channels=in_channels,
            out_channels=hidden_channels,
            heads=heads,
            concat=True,
            dropout=dropout,
            add_self_loops=True,
            **kwargs,
        )
        self.conv2 = GATv2Conv_mask(
            in_channels=hidden_channels * heads,
            out_channels=hidden_channels,
            heads=heads,
            concat=True,
            dropout=dropout,
            add_self_loops=True,
            **kwargs,
        )
        self.conv3 = GATv2Conv_mask(
            in_channels=hidden_channels * heads,
            out_channels=out_channels,
            heads=1,
            concat=False,
            dropout=dropout,
            add_self_loops=True,
            **kwargs,
        )
        self.att = []

    def forward(
        self,
        x,
        edge_index,
        return_att: bool = False,
        mask_edge: Tuple = None,
        post: bool = False,
    ):
        if return_att:
            x, att1 = self.conv1(
                x,
                edge_index,
                return_attention_weights=True,
                mask_edge=mask_edge,
                post=post,
            )
            x = F.elu(x)
            x, att2 = self.conv2(
                x,
                edge_index,
                return_attention_weights=True,
                mask_edge=mask_edge,
                post=post,
            )
            x = F.elu(x)
            x, att3 = self.conv3(
                x,
                edge_index,
                return_attention_weights=True,
                mask_edge=mask_edge,
                post=post,
            )
            self.att = [att1, att2, att3]
        else:
            x = self.conv1(x, edge_index, mask_edge=mask_edge, post=post)
            x = F.elu(x)
            x = self.conv2(x, edge_index, mask_edge=mask_edge, post=post)
            x = F.elu(x)
            x = self.conv3(x, edge_index, mask_edge=mask_edge, post=post)
        return x


class GAT_L3_graph_intervention(torch.nn.Module):
    def __init__(
        self, in_channels, hidden_channels, out_channels, heads, dropout=0, **kwargs
    ):
        super().__init__()
        self.conv1 = GATConv_mask(
            in_channels=in_channels,
            out_channels=hidden_channels,
            heads=heads,
            concat=True,
            dropout=dropout,
            add_self_loops=True,
            **kwargs,
        )
        self.conv2 = GATConv_mask(
            in_channels=hidden_channels * heads,
            out_channels=hidden_channels,
            heads=heads,
            concat=True,
            dropout=dropout,
            add_self_loops=True,
            **kwargs,
        )
        self.conv3 = GATConv_mask(
            in_channels=hidden_channels * heads,
            out_channels=out_channels,
            heads=1,
            concat=False,
            dropout=dropout,
            add_self_loops=True,
            **kwargs,
        )
        self.att = []
        # self.linear = torch.nn.Linear(hidden_channels, out_channels)

    def forward(
        self,
        x,
        edge_index,
        batch: Optional[Tensor] = None,
        return_att: bool = False,
        mask_edge: Tuple = None,
        post: bool = False,
    ):
        if return_att:
            x, att1 = self.conv1(
                x,
                edge_index,
                return_attention_weights=True,
                mask_edge=mask_edge,
                post=post,
            )
            x = F.elu(x)
            x, att2 = self.conv2(
                x,
                edge_index,
                return_attention_weights=True,
                mask_edge=mask_edge,
                post=post,
            )
            x = F.elu(x)
            x, att3 = self.conv3(
                x,
                edge_index,
                return_attention_weights=True,
                mask_edge=mask_edge,
                post=post,
            )
            self.att = [att1, att2, att3]
        else:
            x = self.conv1(x, edge_index, mask_edge=mask_edge, post=post)
            x = F.elu(x)
            x = self.conv2(x, edge_index, mask_edge=mask_edge, post=post)
            x = F.elu(x)
            x = self.conv3(x, edge_index, mask_edge=mask_edge, post=post)

        x = global_add_pool(x, batch)
        # x = self.linear(x)
        return x


# Make a graph classification model based on GATv2_L3_intervention
class GATv2_L3_graph_intervention(torch.nn.Module):
    def __init__(
        self, in_channels, hidden_channels, out_channels, heads, dropout=0, **kwargs
    ):
        super().__init__()
        self.conv1 = GATv2Conv_mask(
            in_channels=in_channels,
            out_channels=hidden_channels,
            heads=heads,
            concat=True,
            dropout=dropout,
            add_self_loops=True,
            **kwargs,
        )
        self.conv2 = GATv2Conv_mask(
            in_channels=hidden_channels * heads,
            out_channels=hidden_channels,
            heads=heads,
            concat=True,
            dropout=dropout,
            add_self_loops=True,
            **kwargs,
        )
        self.conv3 = GATv2Conv_mask(
            in_channels=hidden_channels * heads,
            out_channels=out_channels,
            heads=1,
            concat=False,
            dropout=dropout,
            add_self_loops=True,
            **kwargs,
        )
        self.att = []
        # self.linear = torch.nn.Linear(hidden_channels, out_channels)

    def forward(
        self,
        x,
        edge_index,
        batch: Optional[Tensor] = None,
        return_att: bool = False,
        mask_edge: Tuple = None,
        post: bool = False,
    ):
        if return_att:
            x, att1 = self.conv1(
                x,
                edge_index,
                return_attention_weights=True,
                mask_edge=mask_edge,
                post=post,
            )
            x = F.elu(x)
            x, att2 = self.conv2(
                x,
                edge_index,
                return_attention_weights=True,
                mask_edge=mask_edge,
                post=post,
            )
            x = F.elu(x)
            x, att3 = self.conv3(
                x,
                edge_index,
                return_attention_weights=True,
                mask_edge=mask_edge,
                post=post,
            )
            self.att = [att1, att2, att3]
        else:
            x = self.conv1(x, edge_index, mask_edge=mask_edge, post=post)
            x = F.elu(x)
            x = self.conv2(x, edge_index, mask_edge=mask_edge, post=post)
            x = F.elu(x)
            x = self.conv3(x, edge_index, mask_edge=mask_edge, post=post)

        x = global_add_pool(x, batch)
        # x = self.linear(x)
        return x
