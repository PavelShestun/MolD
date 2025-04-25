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
        self.node_dim = config.get('node_dim', 16)
        
        # Линейное расписание шума
        self.betas = torch.linspace(self.beta_start, self.beta_end, self.num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        
        # Компоненты модели
        self.egcn = EGCN(config)
        self.bond_predictor = BondPredictor(config)
        
        # Модуль для обработки кармана
        self.pocket_encoder = nn.Sequential(
            nn.Linear(self.node_dim + 3, self.config['hidden_dim']),
            nn.ReLU(),
            nn.Linear(self.config['hidden_dim'], self.config['hidden_dim'])
        )
    
    def forward(self, x, coords, pocket_x, pocket_coords, t):
        pocket_features = torch.cat([pocket_x, pocket_coords], dim=-1)
        pocket_embed = self.pocket_encoder(pocket_features)
        denoised_x, denoised_coords, edge_index, edge_attr = self.egcn(x, coords, pocket_x, pocket_coords, t)
        bond_probs = self.bond_predictor(denoised_x, denoised_coords)
        return denoised_x, denoised_coords, bond_probs, edge_index, edge_attr
    
    def add_noise(self, x, coords, t):
        noise = torch.randn_like(x)
        sqrt_alpha_bar = torch.sqrt(self.alpha_bars[t]).view(-1, 1, 1)
        sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alpha_bars[t]).view(-1, 1, 1)
        noisy_x = sqrt_alpha_bar * x + sqrt_one_minus_alpha_bar * noise
        noisy_coords = sqrt_alpha_bar * coords + sqrt_one_minus_alpha_bar * torch.randn_like(coords)
        return noisy_x, noisy_coords, noise
    
    def sample(self, pocket_x, pocket_coords, num_samples, device='cuda', use_ddim=False, ddim_steps=50):
        """
        Генерация молекул с учетом белкового кармана.
        use_ddim: Использовать DDIM для ускоренного сэмплинга.
        ddim_steps: Количество шагов для DDIM (меньше num_timesteps).
        """
        x = torch.randn(num_samples, self.max_nodes, self.node_dim).to(device)
        coords = torch.randn(num_samples, self.max_nodes, 3).to(device)
        pocket_x = pocket_x.to(device)
        pocket_coords = pocket_coords.to(device)
        
        if use_ddim:
            # DDIM: выбор подмножества шагов
            step_indices = torch.linspace(0, self.num_timesteps-1, steps=ddim_steps, dtype=torch.long)
            tau = step_indices.tolist()
            for i in range(len(tau)-1, -1, -1):
                t = torch.full((num_samples,), tau[i], dtype=torch.long, device=device)
                denoised_x, denoised_coords, bond_probs, _, _ = self.forward(
                    x, coords, pocket_x, pocket_coords, t
                )
                
                # DDIM update
                alpha_bar_t = self.alpha_bars[t].view(-1, 1, 1)
                alpha_bar_prev = self.alpha_bars[max(0, t-1)].view(-1, 1, 1) if i > 0 else torch.ones_like(alpha_bar_t)
                sigma = torch.zeros_like(alpha_bar_t)  # Без стохастической компоненты
                pred_x0 = (x - torch.sqrt(1 - alpha_bar_t) * denoised_x) / torch.sqrt(alpha_bar_t)
                x = torch.sqrt(alpha_bar_prev) * pred_x0 + torch.sqrt(1 - alpha_bar_prev - sigma**2) * denoised_x + sigma * torch.randn_like(x)
                
                pred_coords0 = (coords - torch.sqrt(1 - alpha_bar_t) * denoised_coords) / torch.sqrt(alpha_bar_t)
                coords = torch.sqrt(alpha_bar_prev) * pred_coords0 + torch.sqrt(1 - alpha_bar_prev - sigma**2) * denoised_coords + sigma * torch.randn_like(coords)
        else:
            # Стандартный DDPM
            for t in reversed(range(self.num_timesteps)):
                t_tensor = torch.full((num_samples,), t, dtype=torch.long, device=device)
                denoised_x, denoised_coords, bond_probs, _, _ = self.forward(
                    x, coords, pocket_x, pocket_coords, t_tensor
                )
                alpha = self.alphas[t]
                x = (1 / torch.sqrt(alpha)) * (x - (1 - alpha) / torch.sqrt(1 - self.alpha_bars[t]) * denoised_x)
                coords = (1 / torch.sqrt(alpha)) * (coords - (1 - alpha) / torch.sqrt(1 - self.alpha_bars[t]) * denoised_coords)
                if t > 0:
                    x += torch.sqrt(self.betas[t]) * torch.randn_like(x)
                    coords += torch.sqrt(self.betas[t]) * torch.randn_like(coords)
        
        final_bonds = self.bond_predictor(x, coords)
        return x, coords, final_bonds