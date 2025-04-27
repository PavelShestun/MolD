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
        
        self.irreps_scalar = o3.Irreps("16x0e")
        self.irreps_vector = o3.Irreps("16x1o")
        self.irreps_node = self.irreps_scalar + self.irreps_vector
        self.irreps_edge = o3.Irreps("1x0e")  # Исправлено: edge_dim = 1
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
        
        # Исправлено: входной размер = edge_dim (1) + hidden_dim
        self.edge_mlp = FullyConnectedNet(
            [1 + self.hidden_dim, self.hidden_dim, self.hidden_dim],
            act=torch.nn.ReLU()
        )
    
    def build_edges(self, coords):
        batch_size, num_nodes, _ = coords.size()
        dist = torch.norm(coords.unsqueeze(2) - coords.unsqueeze(1), dim=-1)
        edge_index = []
        edge_attr = []
        for b in range(batch_size):
            src, dst = torch.where(dist[b] < self.cutoff)
            edge_index.append(torch.stack([src, dst], dim=0) + b * num_nodes)
            edge_attr.append(dist[b, src, dst].unsqueeze(-1))
        edge_index = torch.cat(edge_index, dim=1)
        edge_attr = torch.cat(edge_attr, dim=0)
        edge_index = to_undirected(edge_index)
        return edge_index, edge_attr
    
    def forward(self, x, coords, pocket_x, pocket_coords, t):
        edge_index, edge_attr = self.build_edges(coords)
        t_embed = self.time_embedding(t.float().unsqueeze(-1))
        node_features = torch.cat([x, coords], dim=-1)
        out = self.propagate(edge_index, x=node_features, edge_attr=edge_attr, t_embed=t_embed)
        denoised_x = out[..., :16]
        denoised_coords = out[..., 16:19]
        return denoised_x, denoised_coords, edge_index, edge_attr
    
    def message(self, x_i, x_j, edge_attr, t_embed):
        """
        x_i, x_j: Признаки узлов для пары рёбер [num_edges, node_dim]
        edge_attr: Признаки рёбер [num_edges, 1]
        t_embed: Эмбеддинг времени [batch_size, hidden_dim]
        """
        num_edges = edge_attr.size(0)
        batch_size = t_embed.size(0)
        t_embed = t_embed.repeat(num_edges // batch_size, 1)  # [num_edges, hidden_dim]
        
        # Проверка размеров
        print(f"edge_attr shape: {edge_attr.shape}, t_embed shape: {t_embed.shape}")
        edge_input = torch.cat([edge_attr, t_embed], dim=-1)
        print(f"edge_input shape: {edge_input.shape}")
        
        edge_features = self.edge_mlp(edge_input)
        msg = self.node_tp(x_i, x_j)
        msg = msg + edge_features
        return msg
    
    def update(self, aggr_out):
        return aggr_out