import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import to_undirected
from e3nn import o3
from e3nn.nn import FullyConnectedNet
from e3nn.o3 import FullyConnectedTensorProduct

class EGCN(MessagePassing):
    def __init__(self, config):
        super(EGCN, self).__init__(aggr='add')
        self.hidden_dim = config.get('hidden_dim', 128)
        self.num_layers = config.get('num_layers', 5)
        self.max_nodes = config.get('max_nodes', 50)
        self.cutoff = config.get('cutoff', 5.0)
        self.node_dim = config.get('node_dim', 16)
        
        # Упрощённые представления для теста
        self.irreps_scalar = o3.Irreps(f"{self.node_dim}x0e")  # 16 скаляров
        self.irreps_vector = o3.Irreps("3x1o")  # 3 вектора (координаты)
        self.irreps_node = self.irreps_scalar + self.irreps_vector
        self.irreps_edge = o3.Irreps("1x0e")
        self.irreps_out = self.irreps_node
        
        self.time_embedding = FullyConnectedNet(
            [1, self.hidden_dim, self.hidden_dim],
            act=torch.nn.ReLU()
        )
        
        self.node_tp = FullyConnectedTensorProduct(
            irreps_in1=self.irreps_node,
            irreps_in2=self.irreps_node,
            irreps_out=self.irreps_out,
            internal_weights=True
        )
        
        self.edge_mlp = FullyConnectedNet(
            [1 + self.hidden_dim, self.hidden_dim, self.hidden_dim],
            act=torch.nn.ReLU()
        )
    
    def build_edges(self, coords, batch_size, num_nodes):
        dist = torch.norm(coords.unsqueeze(2) - coords.unsqueeze(1), dim=-1)
        edge_index = []
        edge_attr = []
        batch = []
        for b in range(batch_size):
            src, dst = torch.where(dist[b] < self.cutoff)
            edge_index.append(torch.stack([src, dst], dim=0))
            edge_attr.append(dist[b, src, dst].unsqueeze(-1))
            batch.append(torch.full_like(src, b))
        edge_index = torch.cat(edge_index, dim=1)
        edge_attr = torch.cat(edge_attr, dim=0)
        batch = torch.cat(batch)
        edge_index = to_undirected(edge_index)
        return edge_index, edge_attr, batch
    
    def forward(self, x, coords, pocket_x, pocket_coords, t):
        batch_size, num_nodes, _ = coords.size()
        edge_index, edge_attr, batch = self.build_edges(coords, batch_size, num_nodes)
        t_embed = self.time_embedding(t.float().unsqueeze(-1))
        node_features = torch.cat([x, coords], dim=-1)  # [batch_size, num_nodes, node_dim + 3]
        out = self.propagate(edge_index, x=node_features, edge_attr=edge_attr, t_embed=t_embed, batch=batch)
        denoised_x = out[..., :self.node_dim]
        denoised_coords = out[..., self.node_dim:self.node_dim + 3]
        return denoised_x, denoised_coords, edge_index, edge_attr
    
    def message(self, x_i, x_j, edge_attr, t_embed, batch):
        num_edges = edge_attr.size(0)
        t_embed = t_embed[batch]
        print(f"x_i shape: {x_i.shape}, x_j shape: {x_j.shape}")
        print(f"edge_attr shape: {edge_attr.shape}, t_embed shape: {t_embed.shape}")
        edge_input = torch.cat([edge_attr, t_embed], dim=-1)
        print(f"edge_input shape: {edge_input.shape}")
        edge_features = self.edge_mlp(edge_input)
        msg = self.node_tp(x_i, x_j)
        msg = msg + edge_features
        return msg
    
    def update(self, aggr_out):
        return aggr_out