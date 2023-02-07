import torch
from torch_geometric.nn.aggr import Set2Set, SumAggregation, MeanAggregation, MaxAggregation, AttentionalAggregation
from torch_geometric.utils import softmax
from conv import GNN_node
from sampler import multinomial_sample_batch
from Emb import full_atom_feature_dims
import torch.nn as nn
from torch_scatter import scatter_max, scatter_add
from set2set import MinDist, MaxCos, SetTransformer
from utils import BatchNorm, MLP


class GNN(nn.Module):

    def __init__(self,
                 num_tasks,
                 num_layer=5,
                 emb_dim=300,
                 norm='gin',
                 virtual_node=True,
                 residual=False,
                 JK="last",
                 graph_pooling="mean",
                 dims=None,
                 outlayer: int = 1,
                 node2nodelayer: int = 0,
                 **kwargs):
        '''
            num_tasks (int): number of labels to be predicted
            virtual_node (bool): whether to add virtual node or not
        '''

        super(GNN, self).__init__()

        self.num_layer = num_layer
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks
        self.graph_pooling = graph_pooling

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        ### GNN to generate node embeddings
        if virtual_node:
            raise NotImplementedError
        else:
            self.gnn_node = GNN_node(num_layer,
                                     emb_dim,
                                     JK=JK,
                                     residual=residual,
                                     norm=norm,
                                     dims=dims,
                                     **kwargs)
        self.node2node = MLP(emb_dim,
                             emb_dim,
                             node2nodelayer,
                             tailact=True,
                             **kwargs["mlp"])
        ### Pooling function to generate whole-graph embeddings
        if self.graph_pooling == "sum":
            self.pool = SumAggregation()
        elif self.graph_pooling == "mean":
            self.pool = MeanAggregation()
        elif self.graph_pooling == "max":
            self.pool = MaxAggregation()
        elif self.graph_pooling == "set2set":
            self.pool = Set2Set(emb_dim, processing_steps=2)
        else:
            raise ValueError("Invalid graph pooling type.")
        outdim = 2 * self.emb_dim if graph_pooling == "set2set" else self.emb_dim
        self.pred_lin = MLP(outdim,
                            num_tasks,
                            outlayer,
                            tailact=False,
                            **kwargs["mlp"])

    def forward(self, batched_data):
        h_node = self.gnn_node(batched_data)
        h_node = self.node2node(h_node)
        h_graph = self.pool(h_node, batched_data.batch)

        return self.pred_lin(h_graph)


class NodeAnchor(GNN):

    def __init__(self,
                 num_tasks,
                 num_layer=5,
                 emb_dim=300,
                 norm='gin',
                 virtual_node=True,
                 residual=False,
                 JK="last",
                 set2set="mindist",
                 **kwargs):
        assert num_tasks == 1
        super().__init__(1, num_layer, emb_dim, norm, virtual_node, residual,
                         JK, "mean", **kwargs)
        if set2set.startswith("id"):
            self.set2set = MLP(emb_dim,
                               emb_dim,
                               0,
                               tailact=False,
                               **kwargs["mlp"])
            self.distlin = nn.Linear(emb_dim, 1)
        elif set2set.startswith("mindist"):
            self.set2set = MinDist()
            self.distlin = nn.Linear(1, 1)
        elif set2set.startswith("maxcos"):
            self.set2set = MaxCos()
            self.distlin = nn.Linear(1, 1)
        else:
            raise NotImplementedError

    def forward(self, batched_data):
        batch = batched_data.batch
        h_node = self.gnn_node(batched_data)
        pred = self.pred_lin(h_node).flatten()
        h_node = self.set2set(h_node, batch)
        pred = self.distlin(h_node).flatten() + pred
        if not self.training:
            return scatter_max(pred, batch)[1], 0, 0
        else:
            prob = softmax(pred, batch)
            logprob = torch.log(prob + 1e-15)
            negentropy = scatter_add(prob * logprob, batch)
            rawsample = multinomial_sample_batch(prob, batch)
            return rawsample, logprob[rawsample], negentropy


class RandomAnchor(nn.Module):

    def __init__(self,
                 num_tasks,
                 num_layer=5,
                 emb_dim=300,
                 norm='gin',
                 virtual_node=True,
                 residual=False,
                 JK="last",
                 set2set="mindist",
                 **kwargs):
        super().__init__()
        assert num_tasks == 1

    def forward(self, batched_data):
        batch = batched_data.batch
        prob = torch.ones_like(batch, dtype=torch.float)
        rawsample = multinomial_sample_batch(prob, batch)
        return rawsample, 0, 0


class AnchorGNN(GNN):

    def __init__(self,
                 num_tasks,
                 num_anchor=1,
                 num_layer=5,
                 emb_dim=300,
                 norm='gin',
                 virtual_node=True,
                 residual=False,
                 JK="last",
                 graph_pooling="mean",
                 an_num_layer=5,
                 an_emb_dim=300,
                 an_norm='gin',
                 an_virtual_node=True,
                 an_residual=False,
                 an_JK="last",
                 an_set2set="id",
                 rand_anchor: bool = False,
                 uni_sample: bool = False,
                 **kwargs):
        super().__init__(num_tasks,
                         num_layer,
                         emb_dim,
                         norm,
                         virtual_node,
                         residual,
                         JK,
                         graph_pooling,
                         dims=full_atom_feature_dims + num_anchor * [2],
                         **kwargs)

        self.num_anchor = num_anchor
        self.anchorsampler = nn.ModuleList()
        sampler_fn = NodeAnchor if not rand_anchor else RandomAnchor
        self.uni_sample = uni_sample
        if uni_sample:
            self.anchorsampler.append(
                sampler_fn(1,
                           an_num_layer,
                           an_emb_dim,
                           an_norm,
                           an_virtual_node,
                           an_residual,
                           an_JK,
                           an_set2set,
                           dims=full_atom_feature_dims + num_anchor * [2],
                           **kwargs))
        for i in range(num_anchor):
            self.anchorsampler.append(
                sampler_fn(1,
                           an_num_layer,
                           an_emb_dim,
                           an_norm,
                           an_virtual_node,
                           an_residual,
                           an_JK,
                           an_set2set,
                           dims=full_atom_feature_dims + i * [2],
                           **kwargs))

    def forward(self, batched_data):
        logprob = torch.tensor(0., device=batched_data.x.device)
        negentropy = torch.tensor(0., device=batched_data.x.device)
        for i in range(self.num_anchor):
            if self.uni_sample:
                rawsample, tlogprob, tnegentropy = self.anchorsampler[0](
                    batched_data)
            else:
                rawsample, tlogprob, tnegentropy = self.anchorsampler[i](
                    batched_data)
            logprob = logprob + tlogprob
            negentropy = negentropy + tnegentropy
            label = torch.zeros_like(batched_data.x[:, [0]])
            label[rawsample] = 1
            batched_data.x = torch.cat((batched_data.x, label), dim=-1)
        h_node = self.gnn_node(batched_data)
        h_node = self.node2node(h_node)
        h_graph = self.pool(h_node, batched_data.batch)
        return self.pred_lin(h_graph), logprob, negentropy


class UniAnchorGNN(GNN):

    def __init__(self,
                 num_tasks,
                 num_anchor=1,
                 num_layer=5,
                 emb_dim=300,
                 norm='gin',
                 virtual_node=True,
                 residual=False,
                 JK="last",
                 graph_pooling="mean",
                 set2set="id",
                 rand_anchor: bool = False,
                 multi_anchor: int = 1,
                 anchor_outlayer: int = 1,
                 outlayer: int = 1,
                 node2nodelayer: int = 1,
                 **kwargs):
        super().__init__(num_tasks,
                         num_layer,
                         emb_dim,
                         norm,
                         virtual_node,
                         residual,
                         JK,
                         graph_pooling,
                         dims=full_atom_feature_dims + num_anchor * [2],
                         node2nodelayer=node2nodelayer,
                         outlayer=outlayer,
                         **kwargs)

        self.num_anchor = num_anchor
        self.multi_anchor = multi_anchor
        self.anchorsampler = nn.ModuleList()
        self.rand_anchor = rand_anchor
        if set2set.startswith("id"):
            self.set2set = MLP(0, 0, 0, tailact=False, **kwargs["mlp"])
            outdim = emb_dim
        elif set2set.startswith("mindist"):
            feat = "feat" in set2set
            concat = "cat" in set2set
            self.set2set = MinDist(feat=feat, concat=concat)
            outdim = (emb_dim if feat else 1) + (emb_dim if concat else 0)
        elif set2set.startswith("maxcos"):
            feat = "feat" in set2set
            concat = "cat" in set2set
            self.set2set = MaxCos(feat=feat, concat=concat)
            outdim = (emb_dim if feat else 1) + (emb_dim if concat else 0)
        else:
            raise NotImplementedError
        self.distlin = MLP(outdim,
                           1,
                           anchor_outlayer,
                           tailact=False,
                           **kwargs["mlp"])
        self.h_node = None

    def get_h_node(self, batched_data):
        self.h_node = self.gnn_node(batched_data)

    def fresh_h_node(self):
        self.h_node = None

    def anchorforward(self, batched_data, T: float = 1):
        assert self.h_node is not None
        batch = batched_data.batch
        if self.rand_anchor:
            prob = torch.ones_like(batch,
                                   dtype=torch.float).unsqueeze_(0).repeat(
                                       self.multi_anchor, 1)
            rawsample = multinomial_sample_batch(prob, batch)
            if self.training:
                return rawsample, 0, 0
            else:
                return rawsample
        h_node = self.set2set(self.h_node, batch)
        pred = self.distlin(h_node).squeeze(-1)
        prob = softmax(pred * T, batch, dim=-1)
        rawsample = multinomial_sample_batch(prob, batch)
        if self.training:
            logprob = torch.log(prob + 1e-15)
            negentropy = scatter_add(prob * logprob, batch, dim=-1)
            return rawsample, torch.gather(logprob, -1, rawsample), negentropy
        else:
            return rawsample

    def graph_forward(self, batched_data):
        assert self.h_node is not None
        h_node = self.node2node(self.h_node)
        h_graph = self.pool(h_node, batched_data.batch)
        return self.pred_lin(h_graph)

    def forward(self, batched_data, T: float = 1):
        self.fresh_h_node()
        if self.training:
            logprob = []
            negentropy = []
            preds = []
            batched_data.x = batched_data.x.unsqueeze(0).repeat(
                self.multi_anchor, 1, 1)
            for i in range(self.num_anchor):
                self.get_h_node(batched_data)
                rawsample, tlogprob, tnegentropy = self.anchorforward(
                    batched_data, T)
                logprob.append(tlogprob)
                negentropy.append(tnegentropy)
                preds.append(self.graph_forward(batched_data))
                self.fresh_h_node()
                label = torch.zeros_like(batched_data.x[:, :, 0])
                label.scatter_(-1, rawsample, 1).unsqueeze_(-1)
                batched_data.x = torch.cat((batched_data.x, label), dim=-1)
            self.get_h_node(batched_data)
            preds.append(self.graph_forward(batched_data))
            return torch.stack(preds, dim=0), torch.stack(
                logprob,
                dim=0), torch.stack(negentropy,
                                    dim=0)  # List[(M, N, 1), (M, N), (M, N)]
        else:
            batched_data.x = batched_data.x.unsqueeze(0).repeat(
                self.multi_anchor, 1, 1)
            for i in range(self.num_anchor):
                self.get_h_node(batched_data)
                rawsample = self.anchorforward(batched_data, T)
                label = torch.zeros_like(batched_data.x[:, :, 0])
                label.scatter_(-1, rawsample, 1).unsqueeze_(-1)
                batched_data.x = torch.cat((batched_data.x, label), dim=-1)
            self.get_h_node(batched_data)
            return self.graph_forward(batched_data).mean(dim=0)


if __name__ == '__main__':
    GNN(num_tasks=10)
