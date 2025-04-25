import torch
import torch.nn as nn
from .egcn import EGCN
from .bond_predictor import BondPredictor

class UEDM(nn.Module):
    def __init__(self, config):
        super(UEDM, self).__init__()
        self.config = config
        self.num_timesteps = config.get('num_timesteps', 1000)
        self.beta_start = config.get('beta_start', 0.0001)
        self.beta_end = config.get('beta_end', 0.02)
        self.max_nodes = config.get('max_nodes', 50)
        self.max_pocket_nodes = config.get('max_pocket_nodes', 100)
        
        # Линейное расписание шума
        self.betas = torch.linspace(self.beta_start, self.beta_end, self.num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        
        # Компоненты модели
        self.egcn = EGCN(config)
        self.bond_predictor = BondPredictor(config)
        
        # Модуль для обработки кармана
        self.pocket_encoder = nn.Sequential(
            nn.Linear(16 + 3, self.config['hidden_dim']),  # Типы атомов + координаты
            nn.ReLU(),
            nn.Linear(self.config['hidden_dim'], self.config['hidden_dim'])
        )
    
    def forward(self, x, coords, pocket_x, pocket_coords, edge_index, edge_attr, t):
        """
        Прямой проход для обучения.
        x: Признаки узлов молекулы [batch_size, num_nodes, node_dim]
        coords: Координаты атомов молекулы [batch_size, num_nodes, 3]
        pocket_x: Признаки узлов кармана [batch_size, num_pocket_nodes, node_dim]
        pocket_coords: Координаты атомов кармана [batch_size, num_pocket_nodes, 3]
        edge_index: Индексы ребер [2, num_edges]
        edge_attr: Признаки ребер [num_edges, edge_dim]
        t: Шаг диффузии [batch_size]
        """
        # Кодирование кармана
        pocket_features = torch.cat([pocket_x, pocket_coords], dim=-1)
        pocket_embed = self.pocket_encoder(pocket_features)  # [batch_size, num_pocket_nodes, hidden_dim]
        
        # Прогон через EGCN с учетом кармана (пока просто передаем признаки молекулы)
        denoised_x, denoised_coords = self.egcn(x, coords, edge_index, edge_attr, t)
        
        # Предсказание связей
        bond_probs = self.bond_predictor(denoised_x, denoised_coords)
        
        return denoised_x, denoised_coords, bond_probs
    
    def add_noise(self, x, coords, t):
        """
        Прямой процесс диффузии: добавление шума.
        """
        noise = torch.randn_like(x)
        sqrt_alpha_bar = torch.sqrt(self.alpha_bars[t]).view(-1, 1, 1)
        sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alpha_bars[t]).view(-1, 1, 1)
        
        noisy_x = sqrt_alpha_bar * x + sqrt_one_minus_alpha_bar * noise
        noisy_coords = sqrt_alpha_bar * coords + sqrt_one_minus_alpha_bar * torch.randn_like(coords)
        
        return noisy_x, noisy_coords, noise
    
    def sample(self, pocket_x, pocket_coords, num_samples, device='cuda'):
        """
        Генерация молекул с учетом белкового кармана.
        pocket_x: Признаки узлов кармана [batch_size, num_pocket_nodes, node_dim]
        pocket_coords: Координаты атомов кармана [batch_size, num_pocket_nodes, 3]
        """
        # Инициализация шума
        x = torch.randn(num_samples, self.max_nodes, self.config['node_dim']).to(device)
        coords = torch.randn(num_samples, self.max_nodes, 3).to(device)
        
        # Подготовка данных кармана
        pocket_x = pocket_x.to(device)
        pocket_coords = pocket_coords.to(device)
        
        # Обратный процесс диффузии
        for t in reversed(range(self.num_timesteps)):
            t_tensor = torch.full((num_samples,), t, dtype=torch.long, device=device)
            denoised_x, denoised_coords, bond_probs = self.forward(
                x, coords, pocket_x, pocket_coords, None, None, t_tensor
            )
            
            # Обновление x и coords
            alpha = self.alphas[t]
            x = (1 / torch.sqrt(alpha)) * (x - (1 - alpha) / torch.sqrt(1 - self.alpha_bars[t]) * denoised_x)
            coords = (1 / torch.sqrt(alpha)) * (coords - (1 - alpha) / torch.sqrt(1 - self.alpha_bars[t]) * denoised_coords)
            
            # Добавление шума на промежуточных шагах
            if t > 0:
                x += torch.sqrt(self.betas[t]) * torch.randn_like(x)
                coords += torch.sqrt(self.betas[t]) * torch.randn_like(coords)
        
        # Финальное предсказание связей
        final_bonds = self.bond_predictor(x, coords)
        
        return x, coords, final_bonds