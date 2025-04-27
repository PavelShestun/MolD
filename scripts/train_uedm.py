import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from models.uedm.diffusion import UEDM
from utils.data_utils import load_data
import yaml
import os

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def train():
    config = load_config('configs/uedm_config.yaml')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UEDM(config['model']['params']).to(device)
    
    dataset = load_data(
        'CrossDocked',
        root=config['data']['root'],
        max_nodes=config['model']['params']['max_nodes'],
        max_pocket_nodes=config['model']['params']['max_pocket_nodes']
    )
    dataloader = DataLoader(dataset, batch_size=config['training']['batch_size'], shuffle=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['lr'])
    criterion_mse = nn.MSELoss()
    criterion_ce = nn.CrossEntropyLoss()
    
    num_epochs = config['training']['num_epochs']
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0
        for batch in dataloader:
            if batch is None:
                continue
            batch = batch.to(device)
            
            try:
                x = batch.x.view(-1, config['model']['params']['max_nodes'], config['model']['params']['node_dim'])
                coords = batch.pos.view(-1, config['model']['params']['max_nodes'], 3)
                pocket_x = batch.pocket_x.view(-1, config['model']['params']['max_pocket_nodes'], config['model']['params']['node_dim'])
                pocket_coords = batch.pocket_pos.view(-1, config['model']['params']['max_pocket_nodes'], 3)
                true_bonds = batch.bonds.view(-1, config['model']['params']['max_nodes'], config['model']['params']['max_nodes'])
                
                # Нормализация данных
                x = (x - x.mean(dim=1, keepdim=True)) / (x.std(dim=1, keepdim=True) + 1e-6)
                coords = coords - coords.mean(dim=1, keepdim=True)
                
                batch_size = x.size(0)
                t = torch.randint(0, model.num_timesteps, (batch_size,), device=device)
                
                noisy_x, noisy_coords, noise = model.add_noise(x, coords, t)
                pred_x, pred_coords, bond_probs, _, _ = model(noisy_x, noisy_coords, pocket_x, pocket_coords, t)
                
                loss_x = criterion_mse(pred_x, x)
                loss_coords = criterion_mse(pred_coords, coords)
                bond_probs_flat = bond_probs.view(-1, 5)
                true_bonds_flat = true_bonds.view(-1)
                loss_bonds = criterion_ce(bond_probs_flat, true_bonds_flat)
                
                loss = loss_x + loss_coords + loss_bonds
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            except Exception as e:
                print(f"=post: Пропуск батча: {str(e)}")
                continue
        
        if num_batches > 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/num_batches:.4f}")
        else:
            print(f"Epoch {epoch+1}/{num_epochs}: Нет валидных батчей")
        
        if (epoch + 1) % config['training']['save_freq'] == 0:
            torch.save(model.state_dict(), f'checkpoints/uedm_epoch_{epoch+1}.pth')

if __name__ == '__main__':
    os.makedirs('checkpoints', exist_ok=True)
    train()