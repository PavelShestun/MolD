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
        
        # Линейное расписание шума (beta)
        self.betas = torch.linspace(self.beta_start, self.beta_end, self.num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        
        # Компоненты модели
        self.egcn = EGCN(config)
        self.bond_predictor = BondPredictor(config)
    
    def forward(self, x, coords, edge_index, edge_attr, t):
        """
        Прямой проход для обучения: предсказание шума.
        x: Признаки узлов (типы атомов)
        coords: Координаты атомов
        edge_index: Индексы ребер
        edge_attr: Признаки ребер
        t: Шаг диффузии
        """
        # Прогон через EGCN для денойзинга
        denoised_x = self.egcn(x, edge_index, edge_attr, t)
        
        # Предсказание связей
        bond_probs = self.bond_predictor(denoised_x, coords)
        
        return denoised_x, bond_probs
    
    def add_noise(self, x, coords, t):
        """
        Прямой процесс диффузии: добавление шума.
        """
        noise = torch.randn_like(x)
        sqrt_alpha_bar = torch.sqrt(self.alpha_bars[t]).view(-1, 1)
        sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alpha_bars[t]).view(-1, 1)
        
        noisy_x = sqrt_alpha_bar * x + sqrt_one_minus_alpha_bar * noise
        noisy_coords = sqrt_alpha_bar * coords + sqrt_one_minus_alpha_bar * torch.randn_like(coords)
        
        return noisy_x, noisy_coords, noise
    
    def sample(self, pocket, num_samples, device='cuda'):
        """
        Генерация молекул с учетом белкового кармана.
        pocket: Данные о белковом кармане (будет реализовано позже)
        """
        # Инициализация случайного шума
        x = torch.randn(num_samples, self.config['max_nodes'], self.config['node_dim']).to(device)
        coords = torch.randn(num_samples, self.config['max_nodes'], 3).to(device)
        
        # Обратный процесс диффузии
        for t in reversed(range(self.num_timesteps)):
            t_tensor = torch.full((num_samples,), t, dtype=torch.long, device=device)
            denoised_x, bond_probs = self.forward(x, coords, None, None, t_tensor)
            
            # Обновление x и coords (упрощенная версия, без DDIM)
            alpha = self.alphas[t]
            x = (1 / torch.sqrt(alpha)) * (x - (1 - alpha) / torch.sqrt(1 - self.alpha_bars[t]) * denoised_x)
            coords = (1 / torch.sqrt(alpha)) * (coords - (1 - alpha) / torch.sqrt(1 - self.alpha_bars[t]) * coords)
            
            # Добавление шума на промежуточных шагах
            if t > 0:
                x += torch.sqrt(self.betas[t]) * torch.randn_like(x)
                coords += torch.sqrt(self.betas[t]) * torch.randn_like(coords)
        
        # Финальное предсказание связей
        final_bonds = self.bond_predictor(x, coords)
        
        return x, coords, final_bonds