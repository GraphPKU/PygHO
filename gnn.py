import torch
from torch_geometric.nn.aggr import Set2Set, SumAggregation, MeanAggregation, MaxAggregation
from torch_geometric.utils import softmax
from torch_geometric.data import Data
from conv import GNN_node
from sampler import multinomial_sample_batch
import torch.nn as nn
from torch_scatter import scatter_max, scatter_add, scatter_min
from set2set import MinDist, MaxCos
from utils import MLP
from Emb import x2dims, MultiEmbedding, SingleEmbedding
from typing import List, Optional
from torch import Tensor
from torch_geometric.utils import degree


class InputEncoder(nn.Module):

    def __init__(self,
                 emb_dim: int,
                 exdims: List[int],
                 zeropad: bool = False,
                 dataset=None,
                 **kwargs) -> None:
        super().__init__()
        if dataset is None:
            self.x_encoder = MultiEmbedding(
                emb_dim,
                dims=exdims,
                lastzeropad=len(exdims) if not zeropad else 0, **kwargs["emb"])
            self.ea_encoder = lambda *args: None
        else:
            x = dataset.data.x
            ea = dataset.data.edge_attr

            if x is None:
                raise NotImplementedError
            elif x.dtype != torch.int64:
                self.x_encoder = MLP(x.shape[-1],
                                     emb_dim,
                                     1,
                                     tailact=True,
                                     **kwargs["mlp"])
            elif x.dtype == torch.int64:
                dims = x2dims(x)
                self.x_encoder = MultiEmbedding(
                    emb_dim,
                    dims=dims + exdims,
                    lastzeropad=len(exdims) if not zeropad else 0, **kwargs["emb"])
            else:
                raise NotImplementedError

            if ea is None:
                self.ea_encoder = lambda *args: None
            elif ea.dtype != torch.int64:
                self.ea_encoder = MLP(ea.shape[-1],
                                      emb_dim,
                                      1,
                                      tailact=True,
                                      **kwargs["mlp"])
            elif ea.dtype == torch.int64:
                dims = x2dims(ea)
                self.ea_encoder = MultiEmbedding(emb_dim, dims=dims, **kwargs["emb"])
            else:
                raise NotImplementedError

    def forward(self, batched_data: Data):
        batched_data.x = self.x_encoder(batched_data.x)
        batched_data.edge_attr = self.ea_encoder(batched_data.edge_attr)
        return batched_data


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
                 outlayer: int = 1,
                 node2nodelayer: int = 0,
                 ln_out: bool=False,
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


        ### GNN to generate node embeddings
        if virtual_node:
            raise NotImplementedError
        else:
            self.gnn_node = GNN_node(num_layer,
                                     emb_dim,
                                     JK=JK,
                                     residual=residual,
                                     norm=norm,
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
        if ln_out:
            print("warning: output is normalized")
        self.pred_lin = nn.Sequential(MLP(outdim,
                            num_tasks,
                            outlayer,
                            tailact=False,
                            **kwargs["mlp"]), nn.LayerNorm(num_tasks, elementwise_affine=False) if ln_out else nn.Identity())

    def forward(self, batched_data):
        h_node = self.gnn_node(batched_data)
        h_node = self.node2node(h_node)
        h_graph = self.pool(h_node, batched_data.batch)

        return self.pred_lin(h_graph)


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
                 policy_detach: bool = False,
                 dataset=None,
                 randinit=False,
                 fullsample=False,
                 nodistlin=False,
                 **kwargs):
        super().__init__(num_tasks,
                         num_layer,
                         emb_dim,
                         norm,
                         virtual_node,
                         residual,
                         JK,
                         graph_pooling,
                         node2nodelayer=node2nodelayer,
                         outlayer=outlayer,
                         **kwargs)
        self.fullsample = fullsample
        if fullsample:
            assert num_anchor < 2, "TODO"
        self.randinit = randinit
        if randinit:
            print("warning: model using random init")
        self.policy_detach = policy_detach
        self.num_anchor = num_anchor
        self.multi_anchor = multi_anchor
        self.rand_anchor = rand_anchor
        self.data_encoder = InputEncoder(emb_dim, [], False, dataset, **kwargs)
        self.anchor_encoder = SingleEmbedding(emb_dim, num_anchor + 1, 1, **kwargs["emb"])
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
        self.distlin = (lambda x: -x) if nodistlin else MLP(outdim,1,anchor_outlayer,tailact=False,tailbias=False,**kwargs["mlp"])
        self.h_node = None

    def get_h_node(self, batched_data):
        self.h_node = self.gnn_node(batched_data)

    def fresh_h_node(self):
        self.h_node = None

    def randomsample(self, batched_data: Data, anchor: Optional[Tensor]=None):
        batch = batched_data.batch
        prob = torch.ones_like(batch, dtype=torch.float).unsqueeze_(0).repeat(self.multi_anchor, 1)
        if anchor is not None:
            prob[anchor > 0] = 0
        rawsample = multinomial_sample_batch(prob, batch)
        return rawsample

    def anchorprob(self, batched_data: Data, anchor: Optional[Tensor]=None, T: float=1):
        assert self.h_node is not None
        batch = batched_data.batch
        if self.policy_detach:
            h_node = self.h_node.detach()
        else:
            h_node = self.h_node
        h_node = self.set2set(h_node, batch)
        print(torch.min(scatter_min(h_node, batch, dim=-2)[0].flatten()))
        pred =  -h_node.squeeze(-1) # self.distlin(h_node).squeeze(-1) #
        if anchor is not None:
            pred[anchor > 0] -= 10  # avoid repeated sample
        prob = softmax(pred * T, batch, dim=-1)
        return prob

    def anchorforward(self,
                      batched_data: Data,
                      T: float = 1,
                      anchor: Optional[Tensor] = None,
                      retlogprob: bool = False):
        batch = batched_data.batch
        if self.rand_anchor:
            rawsample = self.randomsample(batched_data, anchor)
            return rawsample, None, None
        prob = self.anchorprob(batched_data, anchor, T)
        rawsample = multinomial_sample_batch(prob, batch)
        if self.training or retlogprob:
            logprob = torch.log(prob + 1e-15)
            negentropy = scatter_add(prob * logprob, batch, dim=-1)
            return rawsample, torch.gather(logprob, -1, rawsample), negentropy
        else:
            return rawsample, None, None

    def graph_forward(self, batched_data):
        assert self.h_node is not None
        h_node = self.node2node(self.h_node)
        h_graph = self.pool(h_node, batched_data.batch)
        return self.pred_lin(h_graph)

    def preprocessdata(self, batched_data: Data):
        batched_data = self.data_encoder(batched_data)
        batched_data.x = batched_data.x.unsqueeze(0).expand(self.multi_anchor, -1, -1)
        return batched_data

    def addanchor2x(self, batched_data: Data, anchor: Tensor, tx: Tensor):
        batched_data.x = tx * self.anchor_encoder(anchor) + tx
        return batched_data

    def forward(self, batched_data, T: float = 1):
        N = batched_data.y.shape[0]
        anchor = torch.zeros((self.multi_anchor, batched_data.x.shape[-2]),
                             device=batched_data.x.device,
                             dtype=torch.int64)
        batched_data = self.preprocessdata(batched_data)
        if self.randinit:
            batched_data.x = batched_data.x + torch.rand_like(batched_data.x)
        tx = batched_data.x
        logprob = []
        negentropy = []
        preds = []
        if self.fullsample:
            batch = batched_data.batch
            graphsizes = degree(batch, dtype=torch.long, num_nodes=N)
            idx = torch.arange(self.multi_anchor, device=batch.device).unsqueeze_(-1).repeat(1, N)
            mask = (idx > graphsizes)
            offseta = torch.zeros_like(graphsizes)
            torch.cumsum(graphsizes[:-1], dim=0, dtype=torch.long, out=offseta[1:])
            idx += offseta
            idx[mask] = batched_data.x.shape[-2]
            anchor = torch.zeros((self.multi_anchor, batched_data.x.shape[-2]+1),
                         device=batched_data.x.device,
                         dtype=torch.int64)
            anchor.scatter_(-1, idx, 1)
            anchor = anchor[:, :-1]
            batched_data = self.addanchor2x(batched_data, anchor, tx)
        else:
            for i in range(1, self.num_anchor + 1):
                self.get_h_node(batched_data)
                rawsample, tlogprob, tnegentropy = self.anchorforward(
                    batched_data, T, anchor)
                logprob.append(tlogprob)
                negentropy.append(tnegentropy)
                if self.training:
                    preds.append(self.graph_forward(batched_data))
                self.fresh_h_node()
                anchor = anchor.clone()
                anchor.scatter_(-1, rawsample, i)
                batched_data = self.addanchor2x(batched_data, anchor, tx)
        self.get_h_node(batched_data)
        preds.append(self.graph_forward(batched_data))

        finalpred = preds[-1].mean(dim=0)
        if self.training and (self.num_anchor > 0 and not self.fullsample):
            return torch.stack(preds, dim=0), torch.stack(
                logprob, dim=0), torch.stack(negentropy, dim=0), finalpred
        else:
            return torch.stack(preds, dim=0), None, None, finalpred


def softsync(target: nn.Module, source: nn.Module, tau: float):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) +
                                param.data * tau)


def sync(target: nn.Module, source: nn.Module):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


class PPOAnchorGNN(UniAnchorGNN):

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
                 policy_detach: bool = False,
                 dataset=None,
                 tau: float=0.99,
                 **kwargs):
        super().__init__(num_tasks, num_anchor, num_layer, emb_dim, norm,
                         virtual_node, residual, JK, graph_pooling, set2set,
                         rand_anchor, multi_anchor, anchor_outlayer, outlayer,
                         node2nodelayer, policy_detach, dataset, **kwargs)
        self.oldmodel = UniAnchorGNN(num_tasks, num_anchor, num_layer, emb_dim,
                                     norm, virtual_node, residual, JK,
                                     graph_pooling, set2set, rand_anchor,
                                     multi_anchor, anchor_outlayer, outlayer,
                                     node2nodelayer, policy_detach, dataset,
                                     **kwargs)
        sync(self.oldmodel, super())
        self.tau = tau

    def anchorforward(self,
                      batched_data: Data,
                      T: float = 1,
                      anchor: Optional[Tensor] = None,
                      retlogprob: bool = False,
                      rawsample: Optional[Tensor] = None):
        batch = batched_data.batch
        if self.rand_anchor:
            if rawsample is None:
                rawsample = self.randomsample(batched_data, anchor)
            return rawsample, None, None
        prob = self.anchorprob(batched_data, anchor, T)
        if rawsample is None:
            rawsample = multinomial_sample_batch(prob, batch)
        if self.training or retlogprob:
            logprob = torch.log(prob + 1e-15)
            negentropy = scatter_add(prob * logprob, batch, dim=-1)
            return rawsample, torch.gather(logprob, -1, rawsample), negentropy
        else:
            return rawsample, None, None
    
    def updateP(self, tau: Optional[float]=None):
        softsync(self.oldmodel, super(), self.tau if tau is None else tau)
        return

    def forward(self, batched_data: Data, T: float = 1):
        anchor = torch.zeros((self.multi_anchor, batched_data.x.shape[-2]),
                             device=batched_data.x.device,
                             dtype=torch.int64)
        
        old_batched_data = batched_data.clone()
        batched_data = self.preprocessdata(batched_data)
        old_batched_data = self.oldmodel.preprocessdata(old_batched_data)
        tx = batched_data.x
        old_tx = old_batched_data.x
        logprob = []
        oldlogprob = []
        negentropy = []
        preds = []
        for i in range(1, self.num_anchor + 1):
            with torch.no_grad():
                self.oldmodel.eval()
                self.oldmodel.get_h_node(old_batched_data)
                rawsample, oldtlogprob, _ = self.oldmodel.anchorforward(
                    old_batched_data, T, anchor, True)
                self.oldmodel.fresh_h_node()
                oldlogprob.append(oldtlogprob)
            self.get_h_node(batched_data)
            _, tlogprob, tnegentropy = self.anchorforward(batched_data, T, anchor, True, rawsample)
            negentropy.append(tnegentropy)
            logprob.append(tlogprob)
            if self.training:
                preds.append(self.graph_forward(batched_data))
            self.fresh_h_node()
            anchor = anchor.clone()
            anchor.scatter_(-1, rawsample, i)
            batched_data = self.addanchor2x(batched_data, anchor, tx)
            old_batched_data = self.oldmodel.addanchor2x(old_batched_data, anchor, old_tx)
        self.get_h_node(batched_data)
        preds.append(self.graph_forward(batched_data))
        finalpred = preds[-1].mean(dim=0)
        if self.training and (self.num_anchor > 0):
            return torch.stack(preds, dim=0), torch.stack(
                logprob, dim=0), torch.stack(oldlogprob, dim=0), torch.stack(negentropy, dim=0), finalpred
        else:
            return torch.stack(preds, dim=0), None, None, None, finalpred

modeldict = {"policygrad": UniAnchorGNN, "ppo": PPOAnchorGNN}

if __name__ == '__main__':
    GNN(num_tasks=10)
