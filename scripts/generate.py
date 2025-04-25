import torch
from models.uedm.diffusion import UEDM
import yaml
import os

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def generate():
    config = load_config('configs/uedm_config.yaml')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UEDM(config['model']['params']).to(device)
    
    checkpoint_path = 'checkpoints/uedm_epoch_10.pth'
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path))
        print(f"Loaded weights from {checkpoint_path}")
    
    num_samples = 10
    pocket_x = torch.randint(0, 10, (num_samples, config['model']['params']['max_pocket_nodes'], config['model']['params']['node_dim'])).float().to(device)
    pocket_coords = torch.randn(num_samples, config['model']['params']['max_pocket_nodes'], 3).to(device)
    
    model.eval()
    with torch.no_grad():
        x, coords, bond_probs = model.sample(pocket_x, pocket_coords, num_samples, device, use_ddim=True, ddim_steps=50)
    
    bond_types = torch.argmax(bond_probs, dim=-1)
    print("Generated molecules:")
    print(f"x shape: {x.shape}")
    print(f"coords shape: {coords.shape}")
    print(f"bond_types shape: {bond_types.shape}")

if __name__ == '__main__':
    generate()