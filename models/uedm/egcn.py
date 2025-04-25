import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from e3nn import o3
from e3nn.nn import FullyConnectedNet
from e3nn.o3 import FullyConnectedTensorProduct

class EGCN(MessagePassing):
    def __init__(self, config):
        super(EGCN, self).__init__(aggr='add')  # Агрегация сообщений через сумму
        self.hidden_dim = config.get('hidden_dim', 128)
        self.num_layers = config.get('num_layers', 5)
        self.max_nodes = config.get('max_nodes', 50)  # Максимальное число атомов в молекуле
        
        # SE(3)-эквивариантные представления
        self.irreps_node = o3.Irreps("16x0e + 16x1o")  # Скаляры (типы атомов) + векторы (координаты)
        self.irreps_edge = o3.Irreps("8x0e")  # Скалярные признаки ребер (например, расстояния)
        self.irreps_out = o3.Irreps("16x0e + 16x1o")  # Выходные представления
        
        # Эмбеддинги временного шага
        self.time_embedding = nn.Sequential(
            nn.Linear(1, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        
        # Слои для обработки узлов и ребер
        self.node_tp = FullyConnectedTensorProduct(
            irreps_in1=self.irreps_node,
            irreps_in2=self.irreps_node,
            irreps_out=self.irreps_out
        )
        
        self.edge_mlp = FullyConnectedNet(
            [8 + self.hidden_dim, self.hidden_dim, self.hidden_dim],
            act=torch.nn.ReLU()
        )
        
    def forward(self, x, coords, edge_index, edge_attr, t):
        """
        x: Признаки узлов (типы атомов, скалярные признаки) [batch_size, num_nodes, node_dim]
        coords: Координаты атомов [batch_size, num_nodes, 3]
        edge_index: Индексы ребер [2, num_edges]
        edge_attr: Признаки ребер (например, расстояния) [num_edges, edge_dim]
        t: Шаг диффузии [batch_size]
        """
        # Эмбеддинг временного шага
        t_embed = self.time_embedding(t.float().unsqueeze(-1))  # [batch_size, hidden_dim]
        
        # Комбинирование узловых признаков и координат
        node_features = torch.cat([x, coords], dim=-1)  # Временная заглушка, доработка с e3nn позже
        
        # Прямой проход через GNN
        out = self.propagate(edge_index, x=node_features, edge_attr=edge_attr, t_embed=t_embed)
        
        # Разделение выходных признаков на скаляры и векторы
        denoised_x = out[..., :16]  # Скалярные признаки
        denoised_coords = out[..., 16:]  # Векторные признаки (координаты)
        
        return denoised_x, denoised_coords
    
    def message(self, x_i, x_j, edge_attr, t_embed):
        """
        Создание сообщений между узлами.
        x_i, x_j: Признаки соседних узлов
        edge_attr: Признаки ребер
        t_embed: Эмбеддинг временного шага
        """
        # Комбинирование признаков ребер и временного шага
        edge_input = torch.cat([edge_attr, t_embed.repeat(x_i.size(0), 1)], dim=-1)
        edge_features = self.edge_mlp(edge_input)
        
        # SE(3)-эквивариантное сообщение
        msg = self.node_tp(x_i, x_j)
        msg = msg + edge_features  # Упрощенная комбинация
        
        return msg
    
    def update(self, aggr_out):
        """
        Обновление признаков узлов после агрегации сообщений.
        """
        return aggr_out